from playwright.sync_api import sync_playwright

url = "https://www.rightmove.co.uk/properties/137359763"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")

    html = page.content()
    print("analyticsProperty in rendered HTML:",
          "analyticsProperty" in html)
