"""
`tiktok_uploader` module for uploading videos to TikTok

Key Classes
-----------
TikTokUploader : Client for uploading videos to TikTok
"""

import datetime
import logging
import re
import time
from collections.abc import Callable
from os.path import abspath, exists
from typing import Any, Literal

import pytz
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tiktok_uploader import config
from tiktok_uploader.auth import AuthBackend
from tiktok_uploader.browsers import get_browser
from tiktok_uploader.types import Cookie, ProxyDict, VideoDict
from tiktok_uploader.utils import bold, green, red

logger = logging.getLogger(__name__)


class TikTokUploader:
    def __init__(
        self,
        username: str = "",
        password: str = "",
        cookies: str = "",
        cookies_list: list[Cookie] = [],
        cookies_str: str | None = None,
        sessionid: str | None = None,
        proxy: ProxyDict | None = None,
        browser: Literal["chrome", "safari", "chromium", "edge", "firefox"] = "chrome",
        headless: bool = False,
        user_data_dir: str | None = None,
        skip_auth_check: bool = False,  # NEW: skip auth when using existing profile
        *args,
        **kwargs,
    ):
        """
        Initializes the TikTok Uploader client.

        The browser is not started until the first upload is attempted (lazy initialization).

        Args:
            user_data_dir: If provided, browser cookies, cache, and storage will be persisted
                          to this directory and loaded on subsequent runs.
                          If set to None, connects to your global Chrome instance via CDP
                          (Chrome DevTools Protocol on port 9222).
            skip_auth_check: If True and user_data_dir is provided, uses existing browser
                            cookies instead of requiring explicit cookie file.
        """
        self.auth = AuthBackend(
            username=username,
            password=password,
            cookies=cookies,
            cookies_list=cookies_list,
            cookies_str=cookies_str,
            sessionid=sessionid,
            skip_auth_check=skip_auth_check and user_data_dir is not None,
        )
        self.proxy = proxy
        self.browser_name = browser
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.browser_args = args
        self.browser_kwargs = kwargs

        self._page: Page | None = None
        self._browser_context: Any = (
            None  # Stored implicitly via page.context if needed
        )

    @property
    def page(self) -> Page:
        if self._page is None:
            logger.debug(
                "Create a %s browser instance %s",
                self.browser_name,
                "in headless mode" if self.headless else "",
            )
            self._page = get_browser(
                self.browser_name,
                headless=self.headless,
                proxy=self.proxy,
                user_data_dir=self.user_data_dir,
                *self.browser_args,
                **self.browser_kwargs,
            )  # type: ignore[misc]
            self._page = self.auth.authenticate_agent(self._page)
        return self._page

    def upload_video(
        self,
        filename: str,
        description: str = "",
        schedule: datetime.datetime | None = None,
        product_id: str | None = None,
        cover: str | None = None,
        visibility: Literal["everyone", "friends", "only_you"] = "everyone",
        num_retries: int = 1,
        skip_split_window: bool = False,
        skip_interactivity: bool = False,
        *args,
        **kwargs,
    ) -> bool:
        """
        Uploads a single TikTok video.

        Returns True if successful, False otherwise.
        """
        video_dict: VideoDict = {"path": filename}
        if description:
            video_dict["description"] = description
        if schedule:
            video_dict["schedule"] = schedule
        if product_id:
            video_dict["product_id"] = product_id
        if visibility != "everyone":
            video_dict["visibility"] = visibility
        if cover:
            video_dict["cover"] = cover

        failed_list = self.upload_videos(
            [video_dict], num_retries, skip_split_window, skip_interactivity, *args, **kwargs
        )

        return len(failed_list) == 0

    def upload_videos(
        self,
        videos: list[VideoDict],
        num_retries: int = 1,
        skip_split_window: bool = False,
        skip_interactivity: bool = False,
        on_complete: Callable[[VideoDict], None] | None = None,
        *args,
        **kwargs,
    ) -> list[VideoDict]:
        """
        Uploads multiple videos to TikTok.
        Returns a list of failed videos.
        """
        videos = _convert_videos_dict(videos)  # type: ignore

        if videos and len(videos) > 1:
            logger.debug("Uploading %d videos", len(videos))

        page = self.page  # Triggers lazy loading/authentication

        failed = []
        # uploads each video
        for video in videos:
            try:
                path = abspath(video.get("path", "."))
                description = video.get("description", "")
                schedule = video.get("schedule", None)
                product_id = video.get("product_id", None)
                cover_path = video.get("cover", None)
                if cover_path is not None:
                    cover_path = abspath(cover_path)

                visibility = video.get("visibility", "everyone")

                logger.debug(
                    "Posting %s%s",
                    bold(video.get("path", "")),
                    (
                        f"\n{' ' * 15}with description: {bold(description)}"
                        if description
                        else ""
                    ),
                )

                # Video must be of supported type
                if not _check_valid_path(path):
                    print(f"{path} is invalid, skipping")
                    failed.append(video)
                    continue

                # Video must have a valid datetime for tiktok's scheduler
                if schedule:
                    timezone = pytz.UTC
                    if schedule.tzinfo is None:
                        schedule = schedule.astimezone(timezone)
                    elif (utc_offset := schedule.utcoffset()) is not None and int(
                        utc_offset.total_seconds()
                    ) == 0:  # Equivalent to UTC
                        schedule = timezone.localize(schedule)
                    else:
                        print(
                            f"{schedule} is invalid, the schedule datetime must be naive or aware with UTC timezone, skipping"
                        )
                        failed.append(video)
                        continue

                    valid_tiktok_minute_multiple = 5
                    schedule = _get_valid_schedule_minute(
                        schedule, valid_tiktok_minute_multiple
                    )
                    if not _check_valid_schedule(schedule):
                        print(
                            f"{schedule} is invalid, the schedule datetime must be as least 20 minutes in the future, and a maximum of 10 days, skipping"
                        )
                        failed.append(video)
                        continue

                # wrap the whole form sequence in retry logic as well; the
                # ``num_retries`` parameter historically only governed the
                # file-upload step, but navigation issues and other transient
                # problems are best handled by retrying the entire process.
                last_exc = None
                for attempt in range(num_retries):
                    try:
                        complete_upload_form(
                            page,
                            path,
                            description,
                            schedule,
                            skip_split_window,
                            cover_path,
                            product_id,
                            visibility,
                            skip_interactivity,
                            num_retries,
                            self.headless,
                            *args,
                            **kwargs,
                        )  # type: ignore[misc]
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        logger.error(
                            "Attempt %d/%d failed during upload: %s",
                            attempt + 1,
                            num_retries,
                            exc,
                        )
                        # if there are more retries left, try to recover
                        if attempt < num_retries - 1:
                            logger.info(f"Attempting page recovery for attempt {attempt+1}")
                            error_str = str(exc)
                            # If the error is about stale/expired cookies, don't retry - it's hopeless
                            if "stale" in error_str.lower() or "expired" in error_str.lower() or "rejected" in error_str.lower():
                                logger.error(
                                    "Authentication error detected. Retries won't help with invalid cookies."
                                )
                                raise
                            
                            # If the browser context is broken/closed, can't recover in same session
                            if "closed" in error_str.lower() or "context" in error_str.lower():
                                logger.error(
                                    "Browser context is broken. Cannot retry within same session. "
                                    "The outer retry mechanism will create a fresh browser instance."
                                )
                                # Don't try to reload - it will fail. Just raise immediately.
                                raise
                            
                            try:
                                # Try normal reload first
                                page.reload()
                            except Exception as reload_exc:
                                logger.warning(f"Page reload failed: {reload_exc}")
                                # If reload fails, the page context is likely broken
                                # Don't try to create a fresh instance here (causes asyncio issues)
                                # Let the outer retry loop handle it
                                raise
                            time.sleep(2)
                        else:
                            # no attempts left, re-raise the last exception
                            raise
                if last_exc:
                    # this should not normally happen, but guard just in case
                    raise last_exc
            except Exception as exception:
                logger.error("Failed to upload %s", path)
                logger.error(exception)
                failed.append(video)
                # import traceback
                # traceback.print_exc()

            if on_complete and callable(
                on_complete
            ):  # calls the user-specified on-complete function
                on_complete(video)

        return failed

    def close(self):
        """Closes the browser instance and properly cleans up the asyncio event loop."""
        if self._page:
            try:
                self._page.context.browser.close()
            except Exception as e:
                logger.debug(f"Error closing browser: {e}")
            
            # CRITICAL: Stop the PlaywrightSyncAPI instance to prevent asyncio loop leaks
            # This was causing "Playwright Sync API inside asyncio loop" errors on subsequent uses
            try:
                if hasattr(self._page, '_playwright_sync_api'):
                    p = self._page._playwright_sync_api
                    if p is not None:
                        p.stop()
                        logger.debug("PlaywrightSyncAPI stopped successfully")
            except Exception as e:
                logger.debug(f"Error stopping PlaywrightSyncAPI: {e}")
            
            self._page = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Wrapper functions for backward compatibility (optional but good for transition)
def upload_video(
    filename: str,
    description: str | None = None,
    cookies: str = "",
    schedule: datetime.datetime | None = None,
    username: str = "",
    password: str = "",
    sessionid: str | None = None,
    cookies_list: list[Cookie] = [],
    cookies_str: str | None = None,
    proxy: ProxyDict | None = None,
    product_id: str | None = None,
    cover: str | None = None,
    visibility: Literal["everyone", "friends", "only_you"] = "everyone",
    browser: Literal["chrome", "safari", "chromium", "edge", "firefox"] = "chrome",
    headless: bool = False,
    *args,
    **kwargs,
) -> list[VideoDict]:
    """
    Uploads a single TikTok video using the TikTokUploader class.

    Returns a list of failed videos (empty if successful).
    """
    uploader = TikTokUploader(
        username=username,
        password=password,
        cookies=cookies,
        cookies_list=cookies_list,
        cookies_str=cookies_str,
        sessionid=sessionid,
        proxy=proxy,
        browser=browser,
        headless=headless,
        *args,
        **kwargs,
    )  # type: ignore[misc]

    video_dict: VideoDict = {"path": filename}
    if description:
        video_dict["description"] = description
    if schedule:
        video_dict["schedule"] = schedule
    if product_id:
        video_dict["product_id"] = product_id
    if visibility != "everyone":
        video_dict["visibility"] = visibility
    if cover:
        video_dict["cover"] = cover

    try:
        return uploader.upload_videos([video_dict], *args, **kwargs)
    finally:
        if config.quit_on_end:
            uploader.close()


def upload_videos(
    videos: list[VideoDict],
    username: str = "",
    password: str = "",
    cookies: str = "",
    cookies_list: list[Cookie] = [],
    cookies_str: str | None = None,
    sessionid: str | None = None,
    proxy: ProxyDict | None = None,
    browser: Literal["chrome", "safari", "chromium", "edge", "firefox"] = "chrome",
    browser_agent: Page
    | None = None,  # Not fully supported in new class-based approach as constructor
    headless: bool = False,
    *args,
    **kwargs,
) -> list[VideoDict]:
    """
    Uploads multiple videos to TikTok using the TikTokUploader class.
    """
    uploader = TikTokUploader(
        username=username,
        password=password,
        cookies=cookies,
        cookies_list=cookies_list,
        cookies_str=cookies_str,
        sessionid=sessionid,
        proxy=proxy,
        browser=browser,
        headless=headless,
        *args,
        **kwargs,
    )  # type: ignore[misc]

    if browser_agent:
        uploader._page = uploader.auth.authenticate_agent(browser_agent)

    try:
        return uploader.upload_videos(videos, *args, **kwargs)
    finally:
        if config.quit_on_end:
            uploader.close()


def complete_upload_form(
    page: Page,
    path: str,
    description: str,
    schedule: datetime.datetime | None,
    skip_split_window: bool,
    cover_path: str | None = None,
    product_id: str | None = None,
    visibility: Literal["everyone", "friends", "only_you"] = "everyone",
    skip_interactivity: bool = False,
    num_retries: int = 1,
    headless: bool = False,
    *args,
    **kwargs,
) -> None:
    """
    Actually uploads each video
    """
    _go_to_upload(page)

    _click_got_it_if_present(page)
    _remove_cookies_window(page)

    _click_got_it_if_present(page)
    _set_video(page, path=path, num_retries=num_retries, **kwargs)

    if cover_path:
        _click_got_it_if_present(page)
        _set_cover(page, cover_path)
    if not skip_split_window:
        _click_got_it_if_present(page)
        _remove_split_window(page)
    _click_got_it_if_present(page)
    _remove_editing_features_popup(page)
    if not skip_interactivity:
        _click_got_it_if_present(page)
        _set_interactivity(page, **kwargs)
    _click_got_it_if_present(page)
    _set_description(page, description)
    if visibility != "everyone":
        _click_got_it_if_present(page)
        _set_visibility(page, visibility)
    if schedule:
        _click_got_it_if_present(page)
        _set_schedule_video(page, schedule)
    if product_id:
        _click_got_it_if_present(page)
        _add_product_link(page, product_id)
    _click_got_it_if_present(page)
    _post_video(page)


def _click_got_it_if_present(page: Page) -> None:
    """
    Clicks any visible "Got it" button used by TikTok onboarding popups.
    """
    def _click_candidates_once() -> bool:
        clicked = False
        candidates = [
            page.get_by_role("button", name=re.compile(r"^\s*got\s+it\s*!?\s*$", re.IGNORECASE)),
            page.locator("button:has-text('Got it')"),
            page.locator("xpath=//*[@role='button' and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]"),
            page.locator("xpath=//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it') and (self::button or @role='button' or contains(@class, 'button'))]"),
        ]

        for locator in candidates:
            try:
                total = locator.count()
            except Exception:
                continue

            for index in range(total):
                try:
                    button = locator.nth(index)
                    if button.is_visible(timeout=1000):
                        logger.debug(green('Clicking "Got it" popup button'))
                        button.click(timeout=3000, force=True)
                        time.sleep(0.25)
                        clicked = True
                except Exception:
                    continue

        # Some TikTok popups use non-button clickable wrappers.
        try:
            js_clicked = page.evaluate(
                """
                () => {
                    const norm = (s) => (s || '').replace(/\s+/g, ' ').trim().toLowerCase();
                    const labels = new Set(['got it', 'got it!']);
                    const isVisible = (el) => {
                        const st = window.getComputedStyle(el);
                        const r = el.getBoundingClientRect();
                        return st && st.visibility !== 'hidden' && st.display !== 'none' && r.width > 0 && r.height > 0;
                    };

                    const nodes = document.querySelectorAll('button, [role="button"], div, span');
                    for (const el of nodes) {
                        const text = norm(el.textContent);
                        if (!labels.has(text)) continue;
                        if (!isVisible(el)) continue;
                        el.click();
                        return true;
                    }
                    return false;
                }
                """
            )
            if js_clicked:
                logger.debug(green('Clicking "Got it" popup via DOM'))
                time.sleep(0.25)
                clicked = True
        except Exception:
            pass

        return clicked

    # Popup can re-render after animation; retry briefly.
    for _ in range(8):
        if not _click_candidates_once():
            break


def _go_to_upload(page: Page) -> None:
    """
    Navigates to the upload page

    This can be flaky if TikTok momentarily serves a slow or blank page; we
    retry multiple times with improved handling for overlays and redirects.
    """
    logger.debug(green("Navigating to upload page"))

    # Register dialog handler once (prevents duplicate listeners on retries)
    page.once("dialog", lambda dialog: dialog.accept())

    # try up to 3 times with better error handling
    def _wait_for_ready() -> None:
        """Wait for the upload form to be reachable.

        TikTok sometimes renders the page but keeps it hidden behind overlays
        (cookie banners, login prompts, etc). We try multiple strategies:
        1. Wait for #root to be present
        2. Wait for the file-input control directly
        3. Try to dismiss overlays and retry
        """
        # Strategy 1: Wait for #root container
        try:
            page.wait_for_selector("#root", timeout=config.explicit_wait * 1000)
            # Give the page a moment to render
            time.sleep(2)
            _click_got_it_if_present(page)
        except Exception:
            pass

        # Strategy 2: Wait for the file-input control directly
        try:
            _click_got_it_if_present(page)
            page.wait_for_selector(
                f"xpath={config.selectors.upload.upload_video}",
                timeout=config.explicit_wait * 1000,
            )
            return  # Success - file input is ready
        except Exception:
            pass

        # Strategy 3: Check if we're on a redirect/error page and try to recover
        current_url = page.url
        logger.debug(f"Current URL: {current_url}")

        # If we're redirected to login, recover once when session cookie exists.
        if "login" in current_url.lower():
            has_session_cookie = False
            try:
                has_session_cookie = any(
                    c.get("name") in {"sessionid", "sessionid_ss"}
                    for c in page.context.cookies()
                )
            except Exception:
                has_session_cookie = False

            if has_session_cookie:
                logger.warning(
                    "Redirected to login despite session cookie; attempting direct upload page access"
                )
                try:
                    # Wait a moment for any redirects to finish
                    time.sleep(3)
                    
                    # Try to find the file input directly
                    page.wait_for_selector(
                        f"xpath={config.selectors.upload.upload_video}",
                        timeout=config.explicit_wait * 1000,
                    )
                    logger.info("✅ Upload page accessible despite login redirect")
                    return
                except Exception as recovery_exc:
                    error_msg = str(recovery_exc).lower()
                    if "closed" in error_msg or "context" in error_msg:
                        raise Exception(
                            "TikTok rejected your session cookies and the browser session became invalid. "
                            "Your cookies are STALE or EXPIRED. "
                            "\n\nFIX: Ensure you're logged into TikTok in Chrome, then retry."
                        ) from recovery_exc
                    
                    logger.error(f"Direct upload access failed: {recovery_exc}")
                    raise Exception(
                        "Redirected to login despite session cookie and direct access failed. "
                        "Cookie set is likely stale/challenged. "
                        "Ensure you're logged into TikTok in Chrome and retry."
                    ) from recovery_exc

            raise Exception(
                "Redirected to login page. Your TikTok cookies are likely invalid, expired, or challenged. "
                "Please refresh your authentication by running the auth test or updating tiktok_cookies.txt"
            )

        # If we're not on the upload page, something went wrong
        if str(config.paths.upload) not in current_url:
            raise Exception(f"Redirected to {current_url} instead of upload page")

        # The page might have the element but it's hidden behind an overlay
        # Try to click away any overlays
        try:
            _click_got_it_if_present(page)
            page.keyboard.press("Escape")
            time.sleep(1)
        except Exception:
            pass

        # Try one more time for the file input
        try:
            _click_got_it_if_present(page)
            page.wait_for_selector(
                f"xpath={config.selectors.upload.upload_video}",
                timeout=config.explicit_wait * 1000,
            )
        except Exception as final_exc:
            raise Exception(f"Upload page never loaded file input element: {final_exc}") from final_exc

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if str(config.paths.upload) not in page.url:
                page.goto(str(config.paths.upload), wait_until="domcontentloaded", timeout=config.explicit_wait * 1000)
            else:
                # refresh the current upload page
                page.reload(wait_until="domcontentloaded", timeout=config.explicit_wait * 1000)

            # Wait for page to settle (short timeout since TikTok has persistent websockets)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                # domcontentloaded should already be done from goto/reload, this is just a sanity check
                pass

            # waits for the page to be ready for interaction
            _wait_for_ready()
            return
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            logger.debug(f"upload page load attempt {attempt+1} failed: {exc}")
            # If the error is about stale/expired cookies, don't retry
            if ("stale" in error_msg.lower() or "expired" in error_msg.lower() or 
                "rejected" in error_msg.lower() or "session cookies" in error_msg.lower()):
                logger.error("Authentication error detected. No point in retrying navigation.")
                raise  # Re-raise immediately without retry
            if attempt < max_attempts - 1:
                # brief pause before retrying navigation
                logger.info(green(f"Retrying navigation to upload page (attempt {attempt+2}/{max_attempts})"))
                time.sleep(2)
                continue
            # all attempts failed — propagate the error
            raise


def _set_description(page: Page, description: str) -> None:
    """
    Sets the description of the video.

    TikTok will often pre-populate the description field with the name of the
    file that was just uploaded.  if a non-empty string is provided we must
    **first clear any existing text** (including that filename) before typing
    the real description, otherwise the default text remains at the start of
    the caption.  Passing an empty string will still clear the field without
    entering any new content.
    """
    # ``None`` means "do not touch the description at all"; an empty string
    # is handled below so that we can clear auto-filled filenames.
    if description is None:
        return

    logger.debug(green("Setting description"))

    # sanitize description characters
    description = description.encode("utf-8", "ignore").decode("utf-8")
    saved_description = description

    try:
        desc_locator = page.locator(f"xpath={config.selectors.upload.description}")
        desc_locator.wait_for(state="visible", timeout=config.implicit_wait * 1000)

        # Click to focus the field
        desc_locator.click()
        time.sleep(0.5)
        
        # Clear any existing content (filename/previous caption) using keyboard
        _clear(desc_locator)
        time.sleep(0.5)
        
        # Clear again to ensure everything is gone
        _clear(desc_locator)
        time.sleep(0.5)

        # nothing to type if the caller supplied an empty string
        if not description:
            return

        # now enter the provided description word-by-word, handling hashtags
        # and mentions specially as before
        desc_locator.click()
        time.sleep(1)

        words = description.split(" ")
        for word in words:
            if not word:
                continue

            if word[0] == "#":
                desc_locator.press_sequentially(word, delay=50)
                time.sleep(0.5)

                mention_box = page.locator(
                    f"xpath={config.selectors.upload.mention_box}"
                )
                try:
                    mention_box.wait_for(
                        state="visible", timeout=config.add_hashtag_wait * 1000
                    )
                    desc_locator.press("Enter")
                except Exception:
                    pass

            elif word[0] == "@":
                logger.debug(green("- Adding Mention: " + word))
                desc_locator.press_sequentially(word)
                time.sleep(1)

                mention_box_user_id = page.locator(
                    f"xpath={config.selectors.upload.mention_box_user_id}"
                )
                try:
                    mention_box_user_id.first.wait_for(state="visible", timeout=5000)

                    found = False
                    user_ids = mention_box_user_id.all()

                    target_username = word[1:].lower()

                    for i, user_el in enumerate(user_ids):
                        if user_el.is_visible():
                            text = user_el.inner_text().split(" ")[0]
                            if text.lower() == target_username:
                                found = True
                                print("Matching User found : Clicking User")
                                for _ in range(i):
                                    desc_locator.press("ArrowDown")
                                desc_locator.press("Enter")
                                break

                    if not found:
                        desc_locator.press_sequentially(" ")

                except Exception:
                    desc_locator.press_sequentially(" ")

            else:
                desc_locator.press_sequentially(word + " ")

    except Exception as exception:
        print("Failed to set description: ", exception)
        # fallback
        _clear(desc_locator)
        desc_locator.fill(saved_description)


def _clear(locator) -> None:
    """
    Clears the text of the element using platform-appropriate select-all command.
    """
    import platform
    
    # Use Command+A on macOS, Control+A on other platforms
    if platform.system() == "Darwin":  # macOS
        locator.press("Meta+A")
    else:
        locator.press("Control+A")
    
    # Delete the selected text
    locator.press("Backspace")


def _set_video(page: Page, path: str = "", num_retries: int = 3, **kwargs) -> None:
    """
    Sets the video to upload
    """
    logger.debug(green("Uploading video file"))

    for attempt in range(num_retries):
        try:
            # Ensure the upload element is visible and accessible
            upload_box = page.locator(f"xpath={config.selectors.upload.upload_video}")
            
            # Wait for the upload element to be visible before attempting to use it
            try:
                upload_box.wait_for(state="visible", timeout=config.implicit_wait * 1000)
            except Exception:
                logger.warning(f"Upload box not visible on attempt {attempt+1}, trying to recover page")
                # Try pressing Escape to close any overlays
                try:
                    page.keyboard.press("Escape")
                    time.sleep(1)
                except Exception:
                    pass
                # Try waiting again
                upload_box.wait_for(state="visible", timeout=config.implicit_wait * 1000)
            
            # Set the file input
            upload_box.set_input_files(path)
            logger.debug(f"File input set with: {path}")

            # Wait for the video to be processed (look for confirmation element)
            process_confirmation = page.locator(
                f"xpath={config.selectors.upload.process_confirmation}"
            )
            try:
                process_confirmation.wait_for(
                    state="attached", timeout=config.uploading_wait * 1000
                )
                logger.debug("Process confirmation element found - video upload successful")
                return
            except Exception as proc_exc:
                # Check if there's an error message instead
                logger.debug(f"Process confirmation not found: {proc_exc}")
                # Check for error indicators (narrowed to TikTok's actual error UI)
                error_selectors = [
                    "xpath=//div[contains(@class, 'error') and not(contains(@class, 'wrapper'))]",
                    "xpath=//div[@role='alert' or contains(@class, 'error-toast')]",
                    "xpath=//span[contains(text(), 'Upload failed')]",
                ]
                for err_sel in error_selectors:
                    err_elem = page.locator(err_sel)
                    try:
                        err_elem.first.wait_for(state="visible", timeout=2000)
                        error_text = err_elem.first.text_content()
                        raise Exception(f"TikTok reported upload error: {error_text}")
                    except Exception:
                        # Element not visible or doesn't exist, continue checking
                        continue

                # If no error found, maybe the upload is just slow - wait longer
                logger.info("Upload processing... waiting longer")
                try:
                    process_confirmation.wait_for(
                        state="attached", timeout=config.uploading_wait * 1000
                    )
                    logger.debug("Process confirmation found after extended wait - upload successful")
                    return
                except Exception:
                    raise Exception("Video upload timed out - process confirmation not found")
                    
        except PlaywrightTimeoutError as exception:
            logger.error(f"Timeout on attempt {attempt+1}/{num_retries}: {exception}")
            if attempt < num_retries - 1:
                logger.info(f"Retrying upload (attempt {attempt+2}/{num_retries})")
                time.sleep(2)
                # Try to reload the upload page
                try:
                    page.reload(wait_until="domcontentloaded", timeout=config.explicit_wait * 1000)
                    time.sleep(2)
                except Exception:
                    pass
            else:
                raise
        except Exception as exception:
            logger.error(f"Exception on attempt {attempt+1}/{num_retries}: {exception}")
            if attempt < num_retries - 1:
                logger.info(f"Retrying upload (attempt {attempt+2}/{num_retries})")
                time.sleep(2)
            else:
                raise FailedToUpload(exception) from exception


def _remove_cookies_window(page: Page) -> None:
    """
    Removes the cookies window if it is open
    """
    logger.debug(green("Removing cookies window"))

    try:
        selector = f"{config.selectors.upload.cookies_banner.banner} >> {config.selectors.upload.cookies_banner.button} >> button"

        button = page.locator(selector).first
        if button.is_visible(timeout=5000):
            button.click()

    except Exception:
        page.evaluate(f"""
            const banner = document.querySelector("{config.selectors.upload.cookies_banner.banner}");
            if (banner) banner.remove();
        """)


def _remove_split_window(page: Page) -> None:
    """
    Remove the split window if it is open
    """
    logger.debug(green("Removing split window"))
    window_xpath = config.selectors.upload.split_window

    try:
        window = page.locator(f"xpath={window_xpath}")
        if window.is_visible(timeout=config.implicit_wait * 1000):
            window.click()
    except PlaywrightTimeoutError:
        logger.debug(red("Split window not found or operation timed out"))


def _remove_editing_features_popup(page: Page) -> None:
    """
    Remove the editing features popup if it is open
    """
    logger.debug(green("Removing editing features popup"))
    popup_xpath = config.selectors.upload.editing_features_popup

    try:
        button = page.locator(f"xpath={popup_xpath}")
        if button.is_visible(timeout=config.implicit_wait * 1000):
            button.click()
            logger.debug(green("Editing features popup closed"))
    except PlaywrightTimeoutError:
        logger.debug(red("Editing features popup not found or operation timed out"))


def _set_interactivity(
    page: Page,
    comment: bool = True,
    stitch: bool = True,
    duet: bool = True,
    *args,
    **kwargs,
) -> None:
    """
    Sets the interactivity settings of the video
    """
    try:
        logger.debug(green("Setting interactivity settings"))

        comment_box = page.locator(f"xpath={config.selectors.upload.comment}")
        stitch_box = page.locator(f"xpath={config.selectors.upload.stitch}")
        duet_box = page.locator(f"xpath={config.selectors.upload.duet}")

        if comment ^ comment_box.is_checked():
            comment_box.click()

        if stitch ^ stitch_box.is_checked():
            stitch_box.click()

        if duet ^ duet_box.is_checked():
            duet_box.click()

    except Exception as _:
        logger.error("Failed to set interactivity settings")


def _set_visibility(
    page: Page, visibility: Literal["everyone", "friends", "only_you"]
) -> None:
    """
    Sets the visibility/privacy of the video
    """
    try:
        logger.debug(green(f"Setting visibility to: {visibility}"))

        dropdown_xpath = (
            "//div[@data-e2e='video_visibility_container']//button[@role='combobox']"
        )
        dropdown = page.locator(f"xpath={dropdown_xpath}")
        dropdown.click()
        time.sleep(1.5)

        visibility_text_map = {
            "everyone": "Everyone",
            "friends": "Friends",
            "only_you": "Only you",
        }

        option_text = visibility_text_map.get(visibility, "Everyone")
        option_xpath = f"//div[@role='option' and contains(., '{option_text}')]"

        option = page.locator(f"xpath={option_xpath}")
        option.scroll_into_view_if_needed()
        time.sleep(0.5)
        option.click()

        logger.debug(green(f"Successfully set visibility to: {visibility}"))

    except Exception as e:
        logger.error(red(f"Failed to set visibility: {e}"))


def _set_schedule_video(page: Page, schedule: datetime.datetime) -> None:
    """
    Sets the schedule of the video
    """
    logger.debug(green("Setting schedule"))

    timezone_str = page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone")
    driver_timezone = pytz.timezone(timezone_str)

    schedule = schedule.astimezone(driver_timezone)

    month = schedule.month
    day = schedule.day
    hour = schedule.hour
    minute = schedule.minute

    try:
        switch = page.locator(f"xpath={config.selectors.schedule.switch}")
        switch.click()
        __date_picker(page, month, day)
        __time_picker(page, hour, minute)
    except Exception as e:
        msg = f"Failed to set schedule: {e}"
        logger.error(red(msg))
        raise FailedToUpload()


def __date_picker(page: Page, month: int, day: int) -> None:
    logger.debug(green("Picking date"))

    date_picker = page.locator(f"xpath={config.selectors.schedule.date_picker}")
    date_picker.click()

    calendar = page.locator(f"xpath={config.selectors.schedule.calendar}")
    calendar.wait_for(state="visible")

    calendar_month = page.locator(
        f"xpath={config.selectors.schedule.calendar_month}"
    ).inner_text()
    n_calendar_month = datetime.datetime.strptime(calendar_month, "%B").month

    if n_calendar_month != month:
        arrows = page.locator(f"xpath={config.selectors.schedule.calendar_arrows}")
        if n_calendar_month < month:
            arrows.last.click()
        else:
            arrows.first.click()

    valid_days = page.locator(
        f"xpath={config.selectors.schedule.calendar_valid_days}"
    ).all()

    day_to_click = None
    for day_option in valid_days:
        if int(day_option.inner_text()) == day:
            day_to_click = day_option
            break
    if day_to_click:
        day_to_click.click()
    else:
        raise Exception("Day not found in calendar")

    __verify_date_picked_is_correct(page, month, day)


def __verify_date_picked_is_correct(page: Page, month: int, day: int) -> None:
    date_selected = page.locator(
        f"xpath={config.selectors.schedule.date_picker}"
    ).inner_text()
    date_selected_month = int(date_selected.split("-")[1])
    date_selected_day = int(date_selected.split("-")[2])

    if date_selected_month == month and date_selected_day == day:
        logger.debug(green("Date picked correctly"))
    else:
        msg = f"Something went wrong with the date picker, expected {month}-{day} but got {date_selected_month}-{date_selected_day}"
        logger.error(msg)
        raise Exception(msg)


def __time_picker(page: Page, hour: int, minute: int) -> None:
    logger.debug(green("Picking time"))

    time_picker = page.locator(f"xpath={config.selectors.schedule.time_picker}")
    time_picker.click()

    time_picker_container = page.locator(
        f"xpath={config.selectors.schedule.time_picker_container}"
    )
    time_picker_container.wait_for(state="visible")

    hour_options = page.locator(f"xpath={config.selectors.schedule.timepicker_hours}")
    minute_options = page.locator(
        f"xpath={config.selectors.schedule.timepicker_minutes}"
    )

    hour_to_click = hour_options.nth(hour)
    minute_option_correct_index = int(minute / 5)
    minute_to_click = minute_options.nth(minute_option_correct_index)

    hour_to_click.scroll_into_view_if_needed()
    time.sleep(0.5)
    hour_to_click.click()

    minute_to_click.scroll_into_view_if_needed()
    time.sleep(0.5)
    minute_to_click.click()

    time_picker.click()
    time.sleep(0.5)

    __verify_time_picked_is_correct(page, hour, minute)


def __verify_time_picked_is_correct(page: Page, hour: int, minute: int) -> None:
    time_selected = page.locator(
        f"xpath={config.selectors.schedule.time_picker_text}"
    ).inner_text()
    time_selected_hour = int(time_selected.split(":")[0])
    time_selected_minute = int(time_selected.split(":")[1])

    if time_selected_hour == hour and time_selected_minute == minute:
        logger.debug(green("Time picked correctly"))
    else:
        msg = (
            f"Something went wrong with the time picker, "
            f"expected {hour:02d}:{minute:02d} "
            f"but got {time_selected_hour:02d}:{time_selected_minute:02d}"
        )
        raise Exception(msg)


def _post_video(page: Page) -> None:
    """
    Posts the video
    """
    logger.debug(green("Clicking the post button"))

    post_btn = page.locator(f"xpath={config.selectors.upload.post}")
    try:

        def is_enabled():
            return post_btn.get_attribute("data-disabled") == "false"

        for _ in range(int(config.uploading_wait / 2)):
            if is_enabled():
                break
            time.sleep(2)

        post_btn.scroll_into_view_if_needed()
        post_btn.click()

    except Exception:
        logger.debug(green("Trying to click on the button again (fallback)"))
        page.evaluate('document.querySelector(".TUXButton--primary").click()')

    try:
        post_now = page.locator(f"xpath={config.selectors.upload.post_now}")
        if post_now.is_visible(timeout=5000):
            post_now.click()
    except Exception:
        pass

    post_confirmation = page.locator(
        f"xpath={config.selectors.upload.post_confirmation}"
    )
    post_confirmation.wait_for(state="attached", timeout=config.explicit_wait * 1000)

    logger.debug(green("Video posted successfully"))


def _add_product_link(page: Page, product_id: str) -> None:
    """
    Adds the product link
    """
    logger.debug(green(f"Attempting to add product link for ID: {product_id}..."))
    try:
        add_link_button = page.locator(
            "//button[contains(@class, 'Button__root') and contains(., 'Add')]"
        )
        add_link_button.click()
        time.sleep(1)

        try:
            first_next = page.locator(
                "//button[contains(@class, 'TUXButton--primary') and .//div[text()='Next']]"
            )
            if first_next.is_visible(timeout=3000):
                first_next.click()
                time.sleep(1)
        except Exception:
            pass

        search_input = page.locator("//input[@placeholder='Search products']")
        search_input.fill(product_id)
        search_input.press("Enter")
        time.sleep(3)

        product_radio = page.locator(
            f"//tr[.//span[contains(text(), '{product_id}')] or .//div[contains(text(), '{product_id}')]]//input[@type='radio' and contains(@class, 'TUXRadioStandalone-input')]"
        )
        product_radio.click()
        time.sleep(1)

        second_next = page.locator(
            "//button[contains(@class, 'TUXButton--primary') and .//div[text()='Next']]"
        )
        second_next.click()
        time.sleep(1)

        final_add = page.locator(
            "//button[contains(@class, 'TUXButton--primary') and .//div[text()='Add']]"
        )
        final_add.click()

        final_add.wait_for(state="hidden")

    except Exception as e:
        logger.error(red(f"Error adding product link: {e}"))


def _set_cover(page: Page, cover_path: str) -> None:
    """
    Adds a custom cover
    """
    logger.debug(green(f"Attempting to add custom cover: {cover_path}..."))
    try:
        if not _check_valid_cover_path(cover_path):
            raise Exception("Invalid cover image file path")

        preview_loc = page.locator(
            f"xpath={config.selectors.upload.cover.cover_preview}"
        )
        current_cover_src = preview_loc.get_attribute("src")

        edit_cover_btn = page.locator(
            f"xpath={config.selectors.upload.cover.edit_cover_button}"
        )
        edit_cover_btn.click()

        upload_tab = page.locator(
            f"xpath={config.selectors.upload.cover.upload_cover_tab}"
        )
        upload_tab.click()

        upload_box = page.locator(f"xpath={config.selectors.upload.cover.upload_cover}")
        upload_box.set_input_files(cover_path)

        confirm_btn = page.locator(
            f"xpath={config.selectors.upload.cover.upload_confirmation}"
        )
        confirm_btn.click()

        def check_src_change():
            return preview_loc.get_attribute("src") != current_cover_src

        for _ in range(20):
            if check_src_change():
                break
            time.sleep(0.5)

    except Exception as e:
        logger.error(red(f"Error setting cover: {e}"))
        try:
            exit_icon = page.locator(
                f"xpath={config.selectors.upload.cover.exit_cover_container}"
            )
            if exit_icon.is_visible():
                exit_icon.click()
        except Exception:
            pass


def _check_valid_path(path: str) -> bool:
    return exists(path) and path.split(".")[-1] in config.supported_file_types


def _check_valid_cover_path(path: str) -> bool:
    return exists(path) and path.split(".")[-1] in config.supported_image_file_types


def _get_valid_schedule_minute(
    schedule: datetime.datetime, valid_multiple
) -> datetime.datetime:
    if _is_valid_schedule_minute(schedule.minute, valid_multiple):
        return schedule
    else:
        return _set_valid_schedule_minute(schedule, valid_multiple)


def _is_valid_schedule_minute(minute: int, valid_multiple) -> bool:
    if minute % valid_multiple != 0:
        return False
    else:
        return True


def _set_valid_schedule_minute(
    schedule: datetime.datetime, valid_multiple: int
) -> datetime.datetime:
    minute = schedule.minute
    remainder = minute % valid_multiple
    integers_to_valid_multiple = 5 - remainder
    schedule += datetime.timedelta(minutes=integers_to_valid_multiple)
    return schedule


def _check_valid_schedule(schedule: datetime.datetime) -> bool:
    valid_tiktok_minute_multiple = 5
    margin_to_complete_upload_form = 5
    datetime_utc_now = pytz.UTC.localize(datetime.datetime.utcnow())
    min_datetime_tiktok_valid = datetime_utc_now + datetime.timedelta(minutes=15)
    min_datetime_tiktok_valid += datetime.timedelta(
        minutes=margin_to_complete_upload_form
    )
    max_datetime_tiktok_valid = datetime_utc_now + datetime.timedelta(days=10)
    if schedule < min_datetime_tiktok_valid or schedule > max_datetime_tiktok_valid:
        return False
    elif not _is_valid_schedule_minute(schedule.minute, valid_tiktok_minute_multiple):
        return False
    else:
        return True


def _convert_videos_dict(
    videos_list_of_dictionaries: list[dict[str, Any]],
) -> list[VideoDict]:
    if not videos_list_of_dictionaries:
        raise RuntimeError("No videos to upload")

    valid_path = config.valid_path_names
    valid_description = config.valid_descriptions

    correct_path = valid_path[0]
    correct_description = valid_description[0]

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    return_list: list[VideoDict] = []
    for elem in videos_list_of_dictionaries:
        elem = {k.strip().lower(): v for k, v in elem.items()}
        keys = elem.keys()
        path_intersection = intersection(valid_path, keys)
        description_intersection = intersection(valid_description, keys)

        if path_intersection:
            path = elem[path_intersection.pop()]
            if not _check_valid_path(path):
                raise RuntimeError("Invalid path: " + path)
            elem[correct_path] = path
        else:
            for _, value in elem.items():
                if _check_valid_path(value):
                    elem[correct_path] = value
                    break
            else:
                raise RuntimeError("Path not found in dictionary: " + str(elem))

        if description_intersection:
            elem[correct_description] = elem[description_intersection.pop()]
        else:
            for _, value in elem.items():
                if not _check_valid_path(value):
                    elem[correct_description] = value
                    break
            else:
                elem[correct_description] = ""

        return_list.append(elem)  # type: ignore

    return return_list


class DescriptionTooLong(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message or self.__doc__)


class FailedToUpload(Exception):
    def __init__(self, message=None):
        super().__init__(message or self.__doc__)
