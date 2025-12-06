import os
import random
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_POOL = os.getenv("PROXY_POOL", "").split(",")


def get_rotating_proxy():
    """Returns a random full proxy URL from pool."""
    if not USE_PROXY or not PROXY_POOL:
        return None
    return random.choice(PROXY_POOL)


async def scrape_schedule():
    proxy_url = get_rotating_proxy()
    print(f"Using Proxy: {proxy_url}")

    # Convert proxy string into Playwright's proxy format
    proxy_conf = None
    if proxy_url:
        scheme_removed = proxy_url.replace("http://", "")
        user_pass, host_port = scheme_removed.split("@")
        username, password = user_pass.split(":")
        host, port = host_port.split(":")

        proxy_conf = {
            "server": f"http://{host}:{port}",
            "username": username,
            "password": password
        }

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy=proxy_conf
        )
        page = await browser.new_page()

        print("Loading NBA.com/schedule ...")
        await page.goto("https://www.nba.com/schedule", timeout=90000)

        # Wait for schedule section
        await page.wait_for_selector("section.ScheduleMonth_month__2N3mn", timeout=90000)

        print("Page loaded. Extracting games...")

        months = await page.query_selector_all("section.ScheduleMonth_month__2N3mn")

        all_games = []

        for month in months:
            month_name_el = await month.query_selector("h2")
            month_name = await month_name_el.inner_text() if month_name_el else "UNKNOWN"

            days = await month.query_selector_all(".ScheduleDay_day__3k0z6")

            for day in days:
                date_el = await day.query_selector("h3")
                date_text = await date_el.inner_text()

                # Convert "Thu · Dec 4" → "Dec 4 2025"
                date_clean = " ".join(date_text.split("·")[-1].strip().split(" ")[-2:])
                date_full = f"{date_clean} {datetime.now().year}"
                parsed_date = datetime.strptime(date_full, "%b %d %Y")
                parsed_date = parsed_date.strftime("%Y-%m-%d")

                games = await day.query_selector_all(".ScheduleGame_game__2TzCk")

                for g in games:
                    teams = await g.query_selector_all(".ScheduleGame_teamName__1lZEi")
                    if len(teams) < 2:
                        continue

                    away = await teams[0].inner_text()
                    home = await teams[1].inner_text()

                    all_games.append({
                        "date": parsed_date,
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "month": month_name.strip(),
                        "proxy": proxy_url
                    })

        await browser.close()

        df = pd.DataFrame(all_games)
        df = df.sort_values("date")

        # ---- SAVE TO YOUR EXACT PATH ----
        out_path = "/Users/mohamed/Desktop/Projects/NBA_Final/execution/schedule.csv"
        df.to_csv(out_path, index=False)

        print(f"✔ Saved: {out_path}")
        print(df.head(20))


if __name__ == "__main__":
    asyncio.run(scrape_schedule())
