"""Gets the browser's given the user's input"""

import logging
import os
from typing import Any, Literal

from playwright.sync_api import Page, sync_playwright
from playwright_stealth import stealth

from tiktok_uploader import config
from tiktok_uploader.types import ProxyDict

# Type alias for supported browsers
browser_t = Literal["chrome", "firefox", "webkit", "edge", "safari", "chromium"]

logger = logging.getLogger(__name__)


def get_browser(
    name: browser_t = "chrome",
    headless: bool = False,
    proxy: ProxyDict | None = None,
    user_data_dir: str | None = None,
    *args,
    **kwargs,
) -> Page:
    """
    Gets a browser based on the name with the ability to pass in additional arguments
    
    Browser cookies, cache, and all persistent data will be saved to user_data_dir.
    """
    p = sync_playwright().start()

    # Map browser names to Playwright launch functions
    if name == "chrome" or name == "edge" or name == "chromium":
        browser_type = p.chromium
    elif name == "firefox":
        browser_type = p.firefox
    elif name == "webkit" or name == "safari":
        browser_type = p.webkit
    else:
        browser_type = p.chromium  # Default to chromium

    launch_args: dict[str, Any] = {
        "headless": headless,
        "args": [
            "--disable-blink-features=AutomationControlled",
        ],
    }

    if name == "chrome":
        launch_args["channel"] = "chrome"
    elif name == "edge":
        launch_args["channel"] = "msedge"

    if proxy:
        launch_args["proxy"] = {
            "server": f"{proxy['host']}:{proxy['port']}",
        }
        if "user" in proxy and "password" in proxy:
            launch_args["proxy"]["username"] = proxy["user"]
            launch_args["proxy"]["password"] = proxy["password"]

    # Connect to global Chrome instance via CDP when user_data_dir is None
    # This uses your existing browser instead of launching a separate profile
    if user_data_dir is None and name == "chrome":
        # Try to connect to global Chrome via Chrome DevTools Protocol
        cdp_url = "http://127.0.0.1:9222"
        try:
            print(f"\n🔌 Attempting to connect to global Chrome instance at {cdp_url}...")
            print(f"   (Make sure Chrome is started with --remote-debugging-port=9222)")
            browser = browser_type.connect_over_cdp(cdp_url)

            # Use the first existing context (or create one if needed)
            contexts = browser.contexts
            if contexts:
                context = contexts[0]
                page = context.new_page() if len(context.pages) == 0 else context.pages[0]
            else:
                # Create new context if none exist
                context_args = {
                    "viewport": {"width": 1280, "height": 720},
                    "user_agent": config.disguising.user_agent,
                    "locale": "en-US",
                }
                context = browser.new_context(**context_args)
                page = context.new_page()

            page.set_default_timeout(config.implicit_wait * 1000)
            page._playwright_sync_api = p

            print(f"✅ Successfully connected to global Chrome instance\n")
            return page
        except Exception as e:
            print(f"\n⚠️  Could not connect to global Chrome at {cdp_url}")
            print(f"   Error: {e}")
            print(f"   Falling back to launching a new Chrome instance\n")
            # Fall through to normal launch if CDP connection fails

    # Use persistent context if user_data_dir is provided
    # This automatically saves cookies, cache, localStorage, sessionStorage, etc.
    if user_data_dir:
        os.makedirs(user_data_dir, exist_ok=True)

        # Get Chrome's actual version for matching user_agent
        try:
            chrome_version = "146.0.7680.178"  # Current Chrome version
            user_agent = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"
        except:
            user_agent = config.disguising.user_agent

        context_args: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": user_agent,
            "locale": "en-US",
            "timezone_id": "Asia/Jakarta",
            "color_scheme": "light",
        }

        context = browser_type.launch_persistent_context(
            user_data_dir=user_data_dir,
            **launch_args,
            **context_args,
        )

        page = context.new_page() if len(context.pages) == 0 else context.pages[0]
        
        # Apply playwright-stealth anti-detection
        try:
            stealth(page)
            logger.debug("🛡️  Applied playwright-stealth anti-detection")
        except Exception as e:
            logger.warning(f"⚠️  Failed to apply stealth: {e}")
        
        page.set_default_timeout(config.implicit_wait * 1000)  # Convert seconds to ms

        # CRITICAL: Store the PlaywrightSyncAPI instance on the page so it can be
        # properly stopped later. Without this, the asyncio event loop leaks and
        # causes "Playwright Sync API inside asyncio loop" errors on subsequent uses.
        page._playwright_sync_api = p

        return page
    else:
        # Fallback to non-persistent context (data not saved)
        browser = browser_type.launch(**launch_args)

        # Create a new context with stealth-like options
        context_args: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": config.disguising.user_agent,
            "locale": "en-US",
        }

        context = browser.new_context(**context_args)

        # Add init script to mask webdriver
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        page = context.new_page()
        page.set_default_timeout(config.implicit_wait * 1000)  # Convert seconds to ms

        # CRITICAL: Store the PlaywrightSyncAPI instance on the page so it can be
        # properly stopped later. Without this, the asyncio event loop leaks and
        # causes "Playwright Sync API inside asyncio loop" errors on subsequent uses.
        page._playwright_sync_api = p
        
        return page
