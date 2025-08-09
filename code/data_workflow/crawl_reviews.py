import time

from parsel import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize variables
SCROLL_PAUSE_TIME = 3


def extract_reviews(html_content):
    """Extract reviews from Google Maps except reviews that have no text or answer by template

    Args:
        html_content (str): HTML content to extract reviews from

    Returns:
        list: List of reviews. Each review has the form {text, stars, date}
    """
    response = Selector(html_content)
    reviews = []

    for el in response.xpath('//div[contains(@class, "jftiEf fontBodyMedium")]'):
        try:
            review = {
                "text": " ".join(
                    el.xpath(
                        './/div[contains(@class, "MyEned")]//span//text()'
                    ).getall()
                ).strip(),
                "stars": el.xpath(
                    './/span[contains(@class, "kvMYJc")]/@aria-label'
                ).get(""),
                "date": el.xpath('.//span[contains(@class, "rsqaWe")]/text()')
                .get("")
                .strip(),
            }
            if review["text"]:  # Only add if we have review text
                reviews.append(review)
        except Exception as e:
            print(f"Error extracting review: {e}")

    return reviews


def crawl_reviews(
    driver: webdriver.Chrome,
) -> list[dict]:
    """Scroll page and continously collect reviews from Google Maps

    Args:
        driver (webdriver.Chrome): Selenium WebDriver instance

    Returns:
        list: List of of all reviews. Each review has the form {text, stars, date}
    """
    all_reviews = []
    seen_review_hashes = set()
    max_no_new_reviews = 3  # Stop after this many scrolls with no new reviews
    no_new_reviews_count = 0
    # Find the scrollable section
    scrollable_section = driver.find_element(
        By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"
    )
    print("Starting to scroll and collect reviews...")

    while no_new_reviews_count < max_no_new_reviews:
        # Get current page source
        html_content = driver.page_source

        # Extract reviews
        new_reviews = extract_reviews(html_content)

        # Process new reviews
        new_reviews_count = 0
        for review in new_reviews:
            # Create a unique hash for each review
            review_hash = hash(f"{review['text']}{review['date']}")

            if review_hash not in seen_review_hashes:
                seen_review_hashes.add(review_hash)
                all_reviews.append(review)
                new_reviews_count += 1

        print(f"Found {new_reviews_count} new reviews (Total: {len(all_reviews)})")

        # Reset counter if we found new reviews
        if new_reviews_count > 0:
            no_new_reviews_count = 0
        else:
            no_new_reviews_count += 1
            print(
                f"No new reviews found. Attempt {no_new_reviews_count}/{max_no_new_reviews}"
            )

        # Scroll down
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_section
        )
        time.sleep(SCROLL_PAUSE_TIME)

    print(f"\nFinished! Collected {len(all_reviews)} unique reviews")
    return all_reviews
