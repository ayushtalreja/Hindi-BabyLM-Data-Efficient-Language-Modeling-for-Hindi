"""
HTTP utilities for robust web requests with retry logic and rate limiting.

This module provides centralized HTTP client functionality to eliminate duplicate
request handling code across scrapers and downloaders.
"""

import time
import logging
from typing import Optional, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .logging_utils import configure_logger

logger = configure_logger(__name__)


class HTTPClient:
    """Centralized HTTP client with retry logic and rate limiting.

    This class provides a robust HTTP client that handles common scenarios like
    retries, rate limiting, and timeout management, eliminating code duplication
    across different scrapers and downloaders.

    Attributes:
        rate_limit_delay: Delay in seconds between requests
        max_retries: Maximum number of retry attempts
        timeout: Default timeout for requests in seconds
        session: Configured requests.Session instance
    """

    DEFAULT_USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    )

    def __init__(
        self,
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
        timeout: int = 15,
        user_agent: Optional[str] = None
    ):
        """Initialize HTTP client with retry and rate limiting configuration.

        Args:
            rate_limit_delay: Delay in seconds between requests (default: 2.0)
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Default timeout for requests in seconds (default: 15)
            user_agent: Custom user agent string (default: Chrome UA)
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

        # Configure session with retry strategy
        self.session = self._create_session()

        # Track last request time for rate limiting
        self._last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create a configured requests session with retry strategy.

        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Only retry safe methods
        )

        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            'User-Agent': self.user_agent
        })

        return session

    def _apply_rate_limit(self):
        """Apply rate limiting by waiting if needed."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self._last_request_time = time.time()

    def get(
        self,
        url: str,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        apply_rate_limit: bool = True
    ) -> Optional[requests.Response]:
        """Perform GET request with retry logic and rate limiting.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds (default: instance timeout)
            headers: Additional headers to send
            params: URL parameters
            apply_rate_limit: Whether to apply rate limiting (default: True)

        Returns:
            Response object if successful, None if all retries failed

        Examples:
            >>> client = HTTPClient()
            >>> response = client.get('https://example.com')
            >>> if response:
            ...     print(response.text)
        """
        if apply_rate_limit:
            self._apply_rate_limit()

        timeout = timeout or self.timeout

        try:
            logger.debug(f"GET {url}")
            response = self.session.get(
                url,
                timeout=timeout,
                headers=headers,
                params=params
            )

            # Check if request was successful
            if response.status_code == 200:
                return response
            else:
                logger.warning(f"GET {url} returned status {response.status_code}")
                return response  # Return anyway, let caller handle

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching {url}: {e}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_with_custom_retry(
        self,
        url: str,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_on_404: bool = False,
        retry_delay_multiplier: float = 1.0
    ) -> Optional[requests.Response]:
        """Perform GET request with custom retry logic (without session retry).

        This method provides fine-grained control over retry behavior for special cases.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            headers: Additional headers
            retry_on_404: Whether to retry on 404 errors
            retry_delay_multiplier: Multiplier for retry delays

        Returns:
            Response object if successful, None otherwise
        """
        timeout = timeout or self.timeout

        for attempt in range(self.max_retries):
            self._apply_rate_limit()

            try:
                response = requests.get(
                    url,
                    timeout=timeout,
                    headers={**self.session.headers, **(headers or {})},
                )

                if response.status_code == 200:
                    return response

                if response.status_code == 404 and not retry_on_404:
                    logger.warning(f"404 Not Found: {url}")
                    return None

                if response.status_code == 429:  # Too Many Requests
                    wait_time = self.rate_limit_delay * (attempt + 1) * retry_delay_multiplier
                    logger.warning(f"Rate limited (429), waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue

                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: "
                             f"Status {response.status_code} for {url}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

            # Wait before next retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = self.rate_limit_delay * (2 ** attempt) * retry_delay_multiplier
                time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed for {url}")
        return None

    def close(self):
        """Close the session and clean up resources."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for one-off requests
def get_url(
    url: str,
    rate_limit_delay: float = 2.0,
    max_retries: int = 3,
    timeout: int = 15,
    user_agent: Optional[str] = None
) -> Optional[requests.Response]:
    """Convenience function for one-off GET requests.

    Args:
        url: URL to fetch
        rate_limit_delay: Delay between requests
        max_retries: Maximum retry attempts
        timeout: Request timeout
        user_agent: Custom user agent

    Returns:
        Response object if successful, None otherwise

    Examples:
        >>> response = get_url('https://example.com')
        >>> if response:
        ...     print(response.text)
    """
    with HTTPClient(
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries,
        timeout=timeout,
        user_agent=user_agent
    ) as client:
        return client.get(url)
