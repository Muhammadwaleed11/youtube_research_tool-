"""
Utility functions for YouTube Research Tool
Common helper functions used throughout the application
"""

import re
import math
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


# ==================== TIME & DATE UTILITIES ====================

def calculate_vph(views: int, published_at: str) -> float:
    """
    Calculate Views Per Hour (VPH) for a video
    
    Args:
        views: Number of views
        published_at: ISO format timestamp string
    
    Returns:
        Views per hour as float
    """
    try:
        if not views or not published_at:
            return 0.0
        
        # Parse the published timestamp
        # Handle different timestamp formats
        timestamp_str = str(published_at)
        
        # Remove timezone info for simplicity
        if 'T' in timestamp_str:
            # ISO format: 2024-01-01T12:00:00Z
            timestamp_str = timestamp_str.replace('Z', '+00:00')
        
        try:
            if '.' in timestamp_str:
                # Has microseconds
                published_time = datetime.fromisoformat(timestamp_str)
            else:
                # No microseconds
                published_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S%z')
        except:
            # Fallback to simple parsing
            try:
                date_part = timestamp_str.split('T')[0]
                published_time = datetime.strptime(date_part, '%Y-%m-%d')
            except:
                return 0.0
        
        # Current time
        current_time = datetime.now(published_time.tzinfo) if published_time.tzinfo else datetime.utcnow()
        
        # Calculate hours difference
        time_diff = current_time - published_time
        hours_diff = time_diff.total_seconds() / 3600
        
        # Ensure at least 1 hour to avoid division by very small numbers
        hours_diff = max(hours_diff, 1.0)
        
        # Calculate VPH
        vph = views / hours_diff
        
        return float(vph)
    
    except Exception as e:
        logger.error(f"Error calculating VPH: {e}")
        return 0.0


def parse_duration(duration_str: str) -> int:
    """
    Parse ISO 8601 duration string to seconds
    
    Args:
        duration_str: ISO duration string (e.g., PT1H2M3S)
    
    Returns:
        Duration in seconds
    """
    if not duration_str:
        return 0
    
    try:
        # YouTube duration format: PT1H2M3S
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if not match:
            return 0
        
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        return total_seconds
    
    except Exception as e:
        logger.error(f"Error parsing duration {duration_str}: {e}")
        return 0


def human_readable_duration(seconds: int) -> str:
    """
    Convert seconds to human-readable duration
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable string (e.g., "1:23:45" or "12:34")
    """
    if not seconds or seconds <= 0:
        return "0:00"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_timestamp(timestamp: str, format_str: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format timestamp string
    
    Args:
        timestamp: ISO timestamp string
        format_str: Output format
    
    Returns:
        Formatted timestamp string
    """
    try:
        # Clean the timestamp
        ts = str(timestamp).replace('Z', '+00:00')
        
        # Parse
        if '.' in ts:
            dt = datetime.fromisoformat(ts)
        else:
            dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S%z')
        
        # Format
        return dt.strftime(format_str)
    
    except Exception:
        return str(timestamp)


def time_ago(timestamp: str) -> str:
    """
    Convert timestamp to "time ago" format
    
    Args:
        timestamp: ISO timestamp string
    
    Returns:
        Human-readable time ago string
    """
    try:
        # Parse timestamp
        ts = str(timestamp).replace('Z', '+00:00')
        
        if '.' in ts:
            published_time = datetime.fromisoformat(ts)
        else:
            published_time = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S%z')
        
        # Current time
        current_time = datetime.now(published_time.tzinfo) if published_time.tzinfo else datetime.utcnow()
        
        # Calculate difference
        diff = current_time - published_time
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    
    except Exception:
        return "Unknown"


# ==================== NUMBER FORMATTING ====================

def format_number(num: Union[int, float]) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string (e.g., "1.2K", "3.4M")
    """
    try:
        num = float(num)
        
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        elif num == 0:
            return "0"
        elif num < 1:
            return f"{num:.2f}"
        else:
            return f"{int(num):,}"
    
    except Exception:
        return str(num)


def safe_int(value, default: int = 0) -> int:
    """
    Safely convert value to integer
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value
    """
    try:
        if value is None:
            return default
        
        if isinstance(value, str):
            # Remove commas from numbers like "1,234"
            value = value.replace(',', '')
        
        return int(float(value))
    
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value
    """
    try:
        if value is None:
            return default
        
        if isinstance(value, str):
            value = value.replace(',', '')
        
        return float(value)
    
    except Exception:
        return default


def calculate_percentage(part: int, total: int) -> float:
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        Percentage (0-100)
    """
    try:
        if total == 0:
            return 0.0
        
        return (part / total) * 100
    
    except Exception:
        return 0.0


# ==================== DATA PROCESSING ====================

def extract_channel_id_from_url(url: str) -> Optional[str]:
    """
    Extract channel ID from YouTube URL
    
    Args:
        url: YouTube channel URL
    
    Returns:
        Channel ID or None
    """
    patterns = [
        r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
        r'youtube\.com/c/([a-zA-Z0-9_-]+)',
        r'youtube\.com/user/([a-zA-Z0-9_-]+)',
        r'youtube\.com/@([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def extract_video_id_from_url(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL
    
    Args:
        url: YouTube video URL
    
    Returns:
        Video ID or None
    """
    patterns = [
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
        r'youtube\.com/v/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def clean_text(text: str, max_length: int = 500) -> str:
    """
    Clean and truncate text
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(str(text).split())
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned


def calculate_outlier_multiplier(vph_values: List[float]) -> Tuple[float, List[float]]:
    """
    Calculate outlier multipliers for VPH values
    
    Args:
        vph_values: List of VPH values
    
    Returns:
        Tuple of (median_vph, multipliers)
    """
    if not vph_values:
        return 0.0, []
    
    try:
        # Convert to numpy array for calculations
        vph_array = np.array(vph_values, dtype=float)
        
        # Calculate median (more robust than mean for outliers)
        median_vph = np.median(vph_array)
        
        # Avoid division by zero
        if median_vph == 0:
            median_vph = 1.0
        
        # Calculate multipliers
        multipliers = vph_array / median_vph
        
        return float(median_vph), multipliers.tolist()
    
    except Exception as e:
        logger.error(f"Error calculating outlier multipliers: {e}")
        return 0.0, []


def calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
    """
    Calculate engagement rate
    
    Args:
        likes: Number of likes
        comments: Number of comments
        views: Number of views
    
    Returns:
        Engagement rate percentage
    """
    try:
        if views == 0:
            return 0.0
        
        engagement = likes + comments
        rate = (engagement / views) * 100
        
        return round(rate, 2)
    
    except Exception:
        return 0.0


# ==================== DATA EXPORT ====================

def export_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Export DataFrame to CSV
    
    Args:
        df: DataFrame to export
        filename: Output filename
    
    Returns:
        CSV data as string
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_export_{timestamp}.csv"
        
        csv_data = df.to_csv(index=False, encoding='utf-8')
        
        logger.info(f"Exported {len(df)} rows to {filename}")
        return csv_data
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return ""


def export_to_excel(df: pd.DataFrame, filename: str = None) -> bytes:
    """
    Export DataFrame to Excel
    
    Args:
        df: DataFrame to export
        filename: Output filename
    
    Returns:
        Excel data as bytes
    """
    try:
        import io
        from pandas import ExcelWriter
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_export_{timestamp}.xlsx"
        
        output = io.BytesIO()
        
        with ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        
        excel_data = output.getvalue()
        
        logger.info(f"Exported {len(df)} rows to {filename}")
        return excel_data
    
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return b""


# ==================== VALIDATION ====================

def is_valid_youtube_url(url: str) -> bool:
    """
    Check if URL is a valid YouTube URL
    
    Args:
        url: URL to check
    
    Returns:
        True if valid YouTube URL
    """
    youtube_patterns = [
        r'youtube\.com/',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/'
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    
    return False


def is_valid_api_key(api_key: str) -> bool:
    """
    Basic validation for YouTube API key
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if looks like a valid API key
    """
    if not api_key:
        return False
    
    # YouTube API keys are typically 39 characters
    if len(api_key) < 30 or len(api_key) > 50:
        return False
    
    # Should contain alphanumeric characters and underscores
    if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
        return False
    
    return True


# ==================== CACHING ====================

class SimpleCache:
    """
    Simple in-memory cache with TTL (Time To Live)
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
        
        return None
    
    def set(self, key: str, value):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]


# ==================== MISC UTILITIES ====================

def generate_id(prefix: str = "") -> str:
    """
    Generate unique ID
    
    Args:
        prefix: ID prefix
    
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = str(int(time.time() * 1000))[-6:]
    
    return f"{prefix}{timestamp}{random_suffix}"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries, dict2 overwrites dict1
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def flatten_list(nested_list: List) -> List:
    """
    Flatten nested list
    
    Args:
        nested_list: Nested list
    
    Returns:
        Flattened list
    """
    flattened = []
    
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    
    return flattened


# ==================== LEGACY FUNCTIONS (for compatibility) ====================

def calculate_vph_and_outlier(views: int, duration_seconds: int, published_at: str) -> Tuple[float, float]:
    """
    Legacy function for backward compatibility
    
    Args:
        views: Number of views
        duration_seconds: Video duration in seconds
        published_at: Publication timestamp
    
    Returns:
        Tuple of (VPH, outlier_multiplier)
    """
    vph = calculate_vph(views, published_at)
    
    # For compatibility, return 1.0 as default outlier multiplier
    return vph, 1.0


def human_readable_duration_seconds(seconds: int) -> str:
    """Alias for human_readable_duration"""
    return human_readable_duration(seconds)


# Initialize global cache instance
global_cache = SimpleCache(ttl_seconds=300)
