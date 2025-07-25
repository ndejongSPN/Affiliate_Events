"""
Multi‑organization events scraper
================================

This module provides utilities for scraping upcoming events from a handful of
state policy organizations.  Each organization’s website has its own
structure, so bespoke scraping functions live here.  Every function returns
data in a tidy pandas ``DataFrame`` with columns:

    - ``event_name`` (str) – the title of the event
    - ``date`` (datetime.date) – the calendar date of the event
    - ``organization`` (str) – the name of the sponsoring organization

Currently implemented scrapers:

* **AZ Liberty Network** – Uses the "List" view of their WordPress Events
  Calendar to collect upcoming events.
* **Washington Policy Center** – Reads the XML sitemap, fetches all event
  detail pages concurrently and extracts the first date found on each page.
* **Kansas Policy Institute** – Parses the KPI events page for headings and
  adjacent date labels in natural language.
* **Show‑Me Institute** – Utilizes the WordPress "The Events Calendar" REST
  API to gather upcoming events.
* **Nevada Policy** – Uses the same events API as Show‑Me Institute to fetch
  future events.
* **Texas Public Policy Foundation** – Scrapes the foundation’s events page
  for event links and then parses each detail page for a full date.
* **The Buckeye Institute** – Extracts upcoming events from the Buckeye
  Institute’s events listing page.

There are dozens of other think tank websites which may publish event
information, but many employ bespoke CMS platforms or embed events via
third‑party services like Eventbrite.  For those sites this module provides
placeholders to illustrate how additional scrapers can be registered.

The central ``get_all_events()`` function will call each individual scraper
and concatenate their results.  Only events on or after today are returned.

Example usage::

    (.venv)$ python multi_org_event_scraper.py

Inside another script or notebook::

    from multi_org_event_scraper import get_all_events
    df = get_all_events()
    print(df)

Dependencies:
    pip install requests aiohttp beautifulsoup4 python-dateutil pandas pytz

"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

import aiohttp
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup, Tag
from dateutil import parser


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _today(tz: Optional[pytz.timezone] = None) -> dt.date:
    """Return today's date in the given timezone (default system local)."""
    if tz is None:
        return dt.date.today()
    return dt.datetime.now(tz).date()


# ---------------------------------------------------------------------------
# AZ Liberty Network scraper
# ---------------------------------------------------------------------------

_AZ_LIST_URL = "https://azlibertynetwork.org/events/list/"
_AZ_ORG = "AZ Liberty Network"
_AZ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; EventScraper/1.0; +https://example.com)")
}


def _az_extract_events_from_page(soup: BeautifulSoup) -> List[dict]:
    """Return a list of dicts with event_name and date from one AZ list page."""
    rows: List[dict] = []
    seen_titles: set[str] = set()

    # Each event lives in an <article> with class tribe-events-* in the Events
    # Calendar plugin.  Extract the title and a <time> element if present.
    for article in soup.select("article[class*='tribe-events']"):
        title_tag = (
            article.select_one("h3 a")
            or article.select_one("h3")
            or article.find("a", attrs={"data-tribe-event-title": True})
        )
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        # Attempt to parse the machine date from a <time> element
        time_tag = article.find("time")
        raw_dt: Optional[dt.datetime] = None
        if time_tag and time_tag.has_attr("datetime"):
            try:
                raw_dt = parser.isoparse(time_tag["datetime"])
            except (ValueError, TypeError):
                raw_dt = None
        if not raw_dt and time_tag:
            # Fallback: parse the textual content
            try:
                raw_dt = parser.parse(time_tag.get_text(strip=True), fuzzy=True)
            except Exception:
                raw_dt = None
        if not raw_dt:
            # Skip events without a discernible date
            continue

        rows.append(
            {"event_name": title, "date": raw_dt.date(), "organization": _AZ_ORG}
        )
    return rows


def _az_next_page_url(soup: BeautifulSoup) -> Optional[str]:
    """Return the absolute URL of the next list page or None."""
    next_link = soup.find("a", rel="next")
    if next_link and next_link.get("href"):
        return next_link["href"]
    next_link = soup.select_one("a.tribe-events-c-nav__next-link")
    if next_link and next_link.get("href"):
        return next_link["href"]
    return None


def get_az_liberty_events(max_pages: int = 20) -> pd.DataFrame:
    """Scrape upcoming events from AZ Liberty Network."""
    url: Optional[str] = _AZ_LIST_URL
    pages_visited = 0
    all_rows: List[dict] = []
    seen: set[tuple[str, dt.date]] = set()
    today = _today()

    while url and pages_visited < max_pages:
        pages_visited += 1
        resp = requests.get(url, headers=_AZ_HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for row in _az_extract_events_from_page(soup):
            key = (row["event_name"].lower(), row["date"])
            if key in seen or row["date"] < today:
                continue
            seen.add(key)
            all_rows.append(row)

        url = _az_next_page_url(soup)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Washington Policy Center scraper
# ---------------------------------------------------------------------------

_WPC_BASE = "https://www.washingtonpolicy.org"
_WPC_SITEMAP = f"{_WPC_BASE}/sitemap"
_WPC_ORG = "Washington Policy Center"
_WPC_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WPCEventsScraper/2.0)"}

# Regular expression to find dates like MM/DD/YYYY within event pages
_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")


async def _wpc_fetch(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch text content from *url* using aiohttp, handling errors."""
    try:
        async with session.get(
            url, headers=_WPC_HEADERS, timeout=aiohttp.ClientTimeout(total=20)
        ) as r:
            r.raise_for_status()
            return await r.text()
    except Exception:
        return None


def _wpc_extract_first_date(text: str) -> Optional[dt.date]:
    m = _DATE_RE.search(text)
    if m:
        try:
            return parser.parse(m.group(1)).date()
        except parser.ParserError:
            return None
    return None


async def _wpc_parse_event(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    html = await _wpc_fetch(session, url)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1")
    if not title_tag:
        return None
    name = title_tag.get_text(strip=True)
    # Try to find the EVENT INFO section first
    info_hdr = soup.find(string=re.compile(r"^\s*EVENT INFO\s*$", re.I))
    date_val: Optional[dt.date] = None
    if info_hdr and info_hdr.parent:
        block_txt = info_hdr.parent.get_text(" ", strip=True)
        date_val = _wpc_extract_first_date(block_txt)
    if date_val is None:
        date_val = _wpc_extract_first_date(soup.get_text(" ", strip=True))
    return {"event_name": name, "date": date_val, "organization": _WPC_ORG}


async def get_wpc_events_async(concurrency: int = 20) -> pd.DataFrame:
    """Asynchronously scrape upcoming events from Washington Policy Center."""
    async with aiohttp.ClientSession() as session:
        sitemap_xml = await _wpc_fetch(session, _WPC_SITEMAP)
        if not sitemap_xml:
            return pd.DataFrame(columns=["event_name", "date", "organization"])
        locs = re.findall(r"<loc>([^<]+)</loc>", sitemap_xml)
        event_urls = [u for u in locs if "/events/detail/" in u]
        sem = asyncio.Semaphore(concurrency)

        async def sem_task(url: str):
            async with sem:
                return await _wpc_parse_event(session, url)

        tasks = [asyncio.create_task(sem_task(url)) for url in event_urls]
        rows = [row for row in await asyncio.gather(*tasks) if row and row["date"]]
    today = _today()
    df = (
        pd.DataFrame(rows)
        .query("date >= @today")
        .drop_duplicates(subset=["event_name", "date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def get_wpc_events() -> pd.DataFrame:
    """Public wrapper around the async WPC scraper."""
    try:
        return asyncio.run(get_wpc_events_async())
    except Exception:
        # In case the event loop is already running (e.g. inside Jupyter), fall back
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_wpc_events_async())


# ---------------------------------------------------------------------------
# Kansas Policy Institute scraper
# ---------------------------------------------------------------------------

_KPI_URL = "https://kansaspolicy.org/events/"
_KPI_ORG = "Kansas Policy Institute"
_KPI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) "
        "Gecko/20100101 Firefox/124.0"
    )
}

# Regex for Month DD, YYYY (case‑insensitive, variable whitespace)
_KPI_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    flags=re.I,
)


def _kpi_parse_date_string(text: str) -> Optional[dt.date]:
    """Extract Month DD, YYYY from *text* and return a date or None."""
    clean = re.sub(r"^[Dd]ate:\s*", "", text).strip()
    clean = re.split(r"Location:|Show|Register|Join", clean, maxsplit=1, flags=re.I)[0]
    m = _KPI_DATE_RE.search(clean)
    if m:
        try:
            return parser.parse(m.group(0)).date()
        except Exception:
            pass
    try:
        return parser.parse(clean, fuzzy=True).date()
    except (parser.ParserError, ValueError):
        return None


def get_kpi_events(include_past: bool = False, url: str = _KPI_URL) -> pd.DataFrame:
    """Scrape events from Kansas Policy Institute and return a DataFrame."""
    resp = requests.get(url, headers=_KPI_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    records: List[dict] = []
    for h2 in soup.find_all("h2"):
        title = h2.get_text(strip=True)
        if not title or title.lower().startswith("no events"):
            continue
        date_container: Optional[Tag] = h2.find_next(
            lambda tag: isinstance(tag, Tag) and "Date:" in tag.get_text()
        )
        if not date_container:
            continue
        date_text = date_container.get_text(" ", strip=True)
        evt_date = _kpi_parse_date_string(date_text)
        if evt_date is None:
            continue
        records.append(
            {
                "event_name": title,
                "date": evt_date,
                "organization": _KPI_ORG,
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    today = _today(pytz.timezone("America/Chicago"))
    if not include_past:
        df = df[df["date"] >= today]
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Show‑Me Institute scraper (WordPress The Events Calendar API)
# ---------------------------------------------------------------------------


def _get_tribe_events(domain: str, org_name: str, per_page: int = 50) -> pd.DataFrame:
    """Generic helper to fetch upcoming events using the The Events Calendar REST API.

    Parameters
    ----------
    domain : str
        The domain name of the WordPress site (e.g. "showmeinstitute.org").
    org_name : str
        Human readable organization name to attach to each row.
    per_page : int, default 50
        Maximum number of events to request from the API.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["event_name", "date", "organization"].  Rows
        are filtered to ensure only events on or after today are included.
    """
    today = _today()
    # Build the API endpoint; ensure we always use HTTPS.
    url = f"https://{domain}/wp-json/tribe/events/v1/events"
    params = {
        "start_date": today.isoformat(),
        "per_page": per_page,
    }
    headers = {
        # Many sites using The Events Calendar API block generic clients.
        # Supply a browser‑like User‑Agent to reduce the likelihood of a 403.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
    except Exception:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    if resp.status_code != 200:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    try:
        data = resp.json()
    except ValueError:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    events = data.get("events", [])
    rows: List[dict] = []
    for ev in events:
        start_str = ev.get("start_date")
        if not start_str:
            continue
        try:
            start_dt = parser.parse(start_str)
        except Exception:
            continue
        if start_dt.date() < today:
            continue
        title = ev.get("title", "").strip()
        if not title:
            continue
        rows.append(
            {"event_name": title, "date": start_dt.date(), "organization": org_name}
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["event_name", "date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    return df


def get_show_me_institute_events() -> pd.DataFrame:
    """Scrape upcoming events for the Show‑Me Institute."""
    return _get_tribe_events("showmeinstitute.org", "Show‑Me Institute")


def get_nevada_policy_events() -> pd.DataFrame:
    """Scrape upcoming events for the Nevada Policy organization."""
    return _get_tribe_events("nevadapolicy.org", "Nevada Policy")


# ---------------------------------------------------------------------------
# Texas Public Policy Foundation scraper
# ---------------------------------------------------------------------------

_TPPF_EVENTS_URL = "https://www.texaspolicy.com/events/"
_TPPF_ORG = "Texas Public Policy Foundation"
_TPPF_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TPPFEventsScraper/1.0)"}


def _parse_tppf_event_date(text: str) -> Optional[dt.date]:
    """Parse a date string from a Texas Policy event detail page.

    The event pages contain lines like ``Aug 5, 2025 6:00 pm - 8:00 pm`` or
    ``Tuesday, August 5, 2025 6:00 pm - 8:00 pm``.  We look for the first
    occurrence of a month/day/year pattern and return the corresponding date.
    """
    # Match patterns like "Aug 5, 2025" or "November 21-22, 2025"
    pattern = re.compile(r"\b([A-Z][a-z]+\s+\d{1,2}(?:-\d{1,2})?,\s*\d{4})")
    m = pattern.search(text)
    if not m:
        return None
    date_text = m.group(1)
    # Handle ranges such as "Nov 21-22, 2025" by taking the first day
    if "-" in date_text:
        parts = date_text.split("-")
        # The month appears only on the first token (e.g. "Nov 21").  Remove the trailing comma from the last part.
        first_part = parts[0].strip()
        # Append the year from the original match
        year = date_text.split(",")[-1].strip()
        date_candidate = f"{first_part}, {year}"
    else:
        date_candidate = date_text
    try:
        return parser.parse(date_candidate).date()
    except Exception:
        return None


def get_texas_policy_events(max_events: int = 30) -> pd.DataFrame:
    """Scrape upcoming events from the Texas Public Policy Foundation (TPPF).

    This function first parses the events listing page to collect event URLs
    and then visits each detail page to extract the full date.  Only the
    date (not time) is returned in the output.
    """
    today = _today()
    try:
        resp = requests.get(_TPPF_EVENTS_URL, headers=_TPPF_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    soup = BeautifulSoup(resp.text, "html.parser")
    # Collect candidate event URLs from the listing.  Each event is wrapped in an <a> tag
    # inside an <article> on the page.
    urls: List[str] = []
    for article in soup.find_all("article"):
        a_tag = article.find("a", href=True)
        if not a_tag:
            continue
        href: str = a_tag.get("href")
        if not href:
            continue
        if href.endswith("/events/"):
            # Skip the index page
            continue
        if "/events/" not in href:
            continue
        urls.append(href)
    # De‑duplicate and limit
    seen_urls: Set[str] = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen_urls:
            seen_urls.add(u)
            unique_urls.append(u)
    # Limit the number of pages to avoid over‑scraping
    if max_events is not None:
        unique_urls = unique_urls[:max_events]

    rows: List[dict] = []
    for url in unique_urls:
        try:
            r = requests.get(url, headers=_TPPF_HEADERS, timeout=20)
            r.raise_for_status()
        except Exception:
            continue
        page_soup = BeautifulSoup(r.text, "html.parser")
        # Title typically appears in an <h3> or <h1> tag on the detail page
        title_tag = page_soup.find(["h1", "h2", "h3"])
        if not title_tag:
            continue
        name = title_tag.get_text(strip=True)
        # Search for the first date on the page
        date_val = None
        date_val = _parse_tppf_event_date(page_soup.get_text(" ", strip=True))
        if not date_val:
            continue
        if date_val < today:
            continue
        rows.append({"event_name": name, "date": date_val, "organization": _TPPF_ORG})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["event_name", "date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    return df


# ---------------------------------------------------------------------------
# Buckeye Institute scraper
# ---------------------------------------------------------------------------

_BUCKEYE_EVENTS_URL = "https://www.buckeyeinstitute.org/events/"
_BUCKEYE_ORG = "The Buckeye Institute"
_BUCKEYE_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BuckeyeEventsScraper/1.0)"}


def get_buckeye_events() -> pd.DataFrame:
    """Scrape upcoming events from the Buckeye Institute.

    The Buckeye Institute’s events page lists upcoming events with their
    corresponding dates.  This function parses the listing and returns
    events scheduled on or after today.
    """
    today = _today()
    try:
        resp = requests.get(_BUCKEYE_EVENTS_URL, headers=_BUCKEYE_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: List[dict] = []
    # Each event is represented by a div.item__block
    for item in soup.select("div.item__block"):
        title_tag = item.select_one("h3.item__block__title")
        date_span = item.select_one("span.item-date")
        if not title_tag or not date_span:
            continue
        name = title_tag.get_text(strip=True)
        date_text = date_span.get_text(strip=True)
        # Use dateutil parser with fuzzy=True to interpret strings like "Thursday, July 31, 2025"
        try:
            event_date = parser.parse(date_text, fuzzy=True).date()
        except Exception:
            continue
        if event_date < today:
            continue
        rows.append(
            {"event_name": name, "date": event_date, "organization": _BUCKEYE_ORG}
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["event_name", "date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    return df


# ---------------------------------------------------------------------------
# Placeholder for additional organizations
# ---------------------------------------------------------------------------


def get_placeholder_events() -> pd.DataFrame:
    """Return an empty DataFrame for organizations without a scraper.

    This function exists as an example stub for new scrapers.  To add a new
    organization, define a function with signature ``()->pd.DataFrame`` that
    collects event_name, date and organization columns, then register it in
    ``ORGANIZATION_SCRAPERS`` below.
    """
    return pd.DataFrame(columns=["event_name", "date", "organization"])


# Mapping of organization slug to scraping function.  Each function should
# return a DataFrame with the three columns described above.  New entries
# can be added here as more scrapers are written.
ORGANIZATION_SCRAPERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "az_liberty": get_az_liberty_events,
    "wpc": get_wpc_events,
    "kpi": get_kpi_events,
    "show_me_institute": get_show_me_institute_events,
    "nevada_policy": get_nevada_policy_events,
    "texas_policy": get_texas_policy_events,
    "buckeye": get_buckeye_events,
    # Additional scrapers can be registered below.  Functions that return empty
    # DataFrames can be used as placeholders until a bespoke parser is written.
    # For example:
    # "mspc": get_mspc_events,
}


def get_all_events(include_past: bool = False) -> pd.DataFrame:
    """Run all registered scrapers and concatenate their results.

    Parameters
    ----------
    include_past : bool, default **False**
        When False, rows with dates before today are filtered out (if the
        individual scraper does not already filter them).  If True, all
        returned events are included.

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted by date ascending.  Empty if no events were found.
    """
    frames: List[pd.DataFrame] = []
    today = _today()
    for slug, func in ORGANIZATION_SCRAPERS.items():
        try:
            df = func()  # type: ignore
        except TypeError:
            # Some scrapers accept include_past; pass it through
            df = func(include_past=include_past)  # type: ignore
        except Exception as exc:
            print(f"[WARN] {slug} scraper failed: {exc}")
            continue
        if not include_past and not df.empty:
            if "date" in df.columns:
                df = df[df["date"] >= today]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["event_name", "date", "organization"])
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("date").reset_index(drop=True)


def _to_csv_if_requested(df: pd.DataFrame, csv_path: Optional[str]) -> None:
    """Write the DataFrame to *csv_path* if specified and non‑empty."""
    if csv_path:
        out = Path(csv_path)
        df.to_csv(out, index=False)
        print(f"Saved → {out.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser_cli = argparse.ArgumentParser(
        description="Scrape upcoming events across organizations"
    )
    parser_cli.add_argument(
        "--csv", metavar="PATH", type=str, help="Optional CSV output file"
    )
    parser_cli.add_argument(
        "--all", action="store_true", help="Include past events as well"
    )
    args = parser_cli.parse_args()
    df = get_all_events(include_past=args.all)
    pd.set_option("display.max_rows", None)
    print(df)
    site_data_path = Path("site/data.json")
    site_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(site_data_path, orient="records", date_format="iso")
    print(f"Saved → {site_data_path.resolve()}")
