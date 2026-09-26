"""Randomise the parts of an observation that carry no information about the decision.

A page title and a URL say nothing about which operation a field needs, so a model whose answer
depends on them cannot be trusted on a site it has not seen.

Two attempts, because the first was not enough. A pool of twenty plausible titles moved the
acceptance test from 5/10 to 9/10 -- but every one of those ten titles was in the pool, and titles
outside it still answered 7/12. The model had memorised the pool and was treating the title as a
task identifier rather than ignoring it.

So the space has to be large enough that memorisation is impossible: titles are assembled from a
vocabulary, with the separators, casing and length varying too, and URLs are built the same way. The
same reasoning applies to goal slot values -- the city, the date, the count -- which can vary without
changing what is being asked.
"""
from __future__ import annotations

import random

TITLE_WORDS = [
    "Flight", "Flights", "Search", "Book", "Booking", "Order", "Reserve", "Compare", "Deals",
    "Cheap", "Ticket", "Tickets", "Travel", "Trip", "Hotel", "Hotels", "Stay", "Room", "Car",
    "Rental", "Checkout", "Cart", "Payment", "Account", "Sign", "Login", "Register", "Welcome",
    "Home", "Portal", "Dashboard", "Results", "Confirm", "Contact", "Support", "Help", "About",
    "News", "Article", "Report", "Data", "Analysis", "Market", "Finance", "Sports", "Weather",
    "Shop", "Store", "Product", "Category", "Profile", "Settings", "Admin", "Page", "Index",
    "Your", "My", "New", "Quick", "Fast", "Easy", "Best", "Top", "Official", "Online", "Local",
    "Global", "Daily", "Weekly", "China", "Asia", "World", "Plan", "Find", "Get", "Go", "Choose",
    "飞机票", "搜索", "首页", "登录", "注册", "订单", "旅行", "酒店", "新闻", "财经", "体育",
]
TITLE_SEPARATORS = [" - ", " | ", " · ", " — ", " :: ", " "]

HOSTS = ["example.com", "example.org", "shop.example.com", "travel.example.org", "www.example.net",
         "127.0.0.1:8899", "127.0.0.1:8897", "127.0.0.1:8898", "localhost:8000", "app.example.io"]
PATH_WORDS = ["flights", "book", "search", "order", "checkout", "cart", "signup", "login", "list",
              "detail", "results", "confirm", "en", "zh", "home", "index", "p", "v2", "api", "shop",
              "product", "category", "news", "finance", "sports", "hotel", "car", "trip"]
SCHEMES = ["https", "http"]
SUFFIXES = ["", ".html", ".htm", "/", ".php", ".aspx"]


def random_title(rng: random.Random) -> str:
    """A fresh title nearly every call: 1-4 words, varying case, separators and occasional noise."""
    roll = rng.random()
    if roll < 0.04:
        return ""
    if roll < 0.08:
        return rng.choice(["首页", "登录", "搜索结果", "订单确认"])
    parts = [rng.choice(TITLE_WORDS) for _ in range(rng.randint(1, 4))]
    title = rng.choice(TITLE_SEPARATORS).join(parts)
    style = rng.random()
    if style < 0.20:
        title = title.title()
    elif style < 0.35:
        title = title.lower()
    elif style < 0.45:
        title = title.upper()
    if rng.random() < 0.12:
        title += f" {rng.randint(1, 99)}"
    if rng.random() < 0.06:
        title = "..." + title
    return title


def random_url(rng: random.Random) -> str:
    host = rng.choice(HOSTS)
    path = "/".join(rng.choice(PATH_WORDS) for _ in range(rng.randint(0, 3)))
    url = f"{rng.choice(SCHEMES)}://{host}"
    if path:
        url += "/" + path
    url += rng.choice(SUFFIXES)
    if rng.random() < 0.20:
        url += f"?{'&'.join(f'{rng.choice(PATH_WORDS)}={rng.randint(1, 999)}' for _ in range(rng.randint(1, 3)))}"
    return url


def randomise_surface(prompt: str, rng: random.Random) -> str:
    """Rewrite the `Page: <title>  (<url>)` line, leaving every other line byte for byte."""
    lines = prompt.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Page: "):
            lines[i] = f"Page: {random_title(rng)}  ({random_url(rng)})"
            break
    return "\n".join(lines)
