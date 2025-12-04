"""
YouTube Data API v3 wrapper for YouTube Research Tool
Handles all YouTube API calls with rate limiting and error handling
"""

import time
import re
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logger = logging.getLogger("youtube_api")
logger.setLevel(logging.INFO)


class YoutubeAPIError(Exception):
    """Custom exception for YouTube API errors"""
    pass


def rate_limited(min_interval_seconds: float = 0.2):
    """
    Rate limiting decorator to avoid hitting API limits
    """
    def decorator(func):
        last_called_attr = f"_last_called_{func.__name__}"
        
        def wrapper(self, *args, **kwargs):
            last_called = getattr(self, last_called_attr, 0)
            elapsed = time.time() - last_called
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
            
            result = func(self, *args, **kwargs)
            setattr(self, last_called_attr, time.time())
            return result
        
        return wrapper
    
    return decorator


def iso8601_duration_to_seconds(duration: str) -> int:
    """
    Convert ISO 8601 duration string to seconds
    Example: "PT1H2M3S" -> 3723 seconds
    """
    if not duration:
        return 0
    
    pattern = re.compile(
        r'P'
        r'(?:(?P<days>\d+)D)?'
        r'(?:T'
        r'(?:(?P<hours>\d+)H)?'
        r'(?:(?P<minutes>\d+)M)?'
        r'(?:(?P<seconds>\d+)S)?'
        r')?'
    )
    
    match = pattern.match(duration)
    if not match:
        return 0
    
    parts = match.groupdict()
    days = int(parts['days']) if parts['days'] else 0
    hours = int(parts['hours']) if parts['hours'] else 0
    minutes = int(parts['minutes']) if parts['minutes'] else 0
    seconds = int(parts['seconds']) if parts['seconds'] else 0
    
    return (days * 86400) + (hours * 3600) + (minutes * 60) + seconds


class YouTubeAPI:
    """
    Main YouTube API wrapper class
    """
    
    def __init__(self, api_key: str, requests_per_second: float = 5.0, max_retries: int = 3):
        """
        Initialize YouTube API client
        
        Args:
            api_key: YouTube Data API v3 key
            requests_per_second: Rate limit (default: 5 requests/second)
            max_retries: Maximum retry attempts for failed requests
        """
        if not api_key:
            raise YoutubeAPIError("API key is required")
        
        self.api_key = api_key
        self.max_retries = max_retries
        
        try:
            self.service = build('youtube', 'v3', developerKey=api_key)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            raise YoutubeAPIError(f"Failed to initialize YouTube client: {e}")
    
    @rate_limited(min_interval_seconds=0.2)
    def _call_api(self, resource_callable, **kwargs):
        """
        Low-level API call with retry logic
        """
        for attempt in range(self.max_retries):
            try:
                request = resource_callable(**kwargs)
                response = request.execute()
                return response
            
            except HttpError as e:
                status = e.resp.status if hasattr(e, 'resp') else None
                
                # If quota exceeded or too many requests, wait and retry
                if status in [429, 403] and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff
                    logger.warning(f"Rate limited (status {status}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # For other HTTP errors, raise immediately
                raise YoutubeAPIError(f"YouTube API HTTP error {status}: {e}")
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise YoutubeAPIError(f"YouTube API error: {e}")
    
    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        published_after: Optional[str] = None,
        duration: Optional[str] = None,
        include_live: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube videos with given parameters
        
        Args:
            query: Search keywords
            max_results: Maximum number of results (max 50 per page)
            published_after: RFC3339 timestamp for filtering
            duration: 'short', 'medium', 'long', or None
            include_live: Include live streams
        
        Returns:
            List of video dictionaries with metadata
        """
        if not query:
            raise YoutubeAPIError("Search query is required")
        
        max_results = min(max_results, 200)  # Cap at 200
        results = []
        next_page_token = None
        
        try:
            while len(results) < max_results:
                # Prepare search parameters
                search_params = {
                    'part': 'snippet',
                    'q': query,
                    'type': 'video',
                    'maxResults': min(50, max_results - len(results)),
                    'order': 'viewCount'
                }
                
                if published_after:
                    search_params['publishedAfter'] = published_after
                
                if duration:
                    search_params['videoDuration'] = duration
                
                if not include_live:
                    search_params['eventType'] = 'completed'
                
                if next_page_token:
                    search_params['pageToken'] = next_page_token
                
                # Execute search
                search_response = self._call_api(self.service.search().list, **search_params)
                search_items = search_response.get('items', [])
                
                if not search_items:
                    break
                
                # Get video IDs
                video_ids = [item['id']['videoId'] for item in search_items if 'videoId' in item['id']]
                
                if not video_ids:
                    continue
                
                # Get video details
                videos_response = self._call_api(
                    self.service.videos().list,
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids)
                )
                
                video_items = videos_response.get('items', [])
                
                # Process each video
                for video in video_items:
                    video_id = video['id']
                    snippet = video.get('snippet', {})
                    statistics = video.get('statistics', {})
                    content_details = video.get('contentDetails', {})
                    
                    # Get channel details
                    channel_id = snippet.get('channelId')
                    channel_stats = {}
                    
                    if channel_id:
                        try:
                            channel_response = self._call_api(
                                self.service.channels().list,
                                part='statistics,snippet',
                                id=channel_id
                            )
                            if channel_response.get('items'):
                                channel_stats = channel_response['items'][0].get('statistics', {})
                        except Exception:
                            pass  # Channel stats are optional
                    
                    # Build result dictionary
                    result = {
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', ''),
                        'channel_id': channel_id,
                        'channel_title': snippet.get('channelTitle', ''),
                        'channel_subscribers': int(channel_stats.get('subscriberCount', 0)) if channel_stats.get('subscriberCount') else None,
                        'views': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                        'likes': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else 0,
                        'comments': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else 0,
                        'duration_seconds': iso8601_duration_to_seconds(content_details.get('duration', '')),
                        'published_at': snippet.get('publishedAt', ''),
                        'thumbnail_url': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                        'url': f'https://www.youtube.com/watch?v={video_id}'
                    }
                    
                    results.append(result)
                
                # Check for more pages
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token or len(results) >= max_results:
                    break
            
            logger.info(f"Found {len(results)} videos for query: '{query}'")
            return results
        
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise YoutubeAPIError(f"Search failed: {e}")
    
    def get_channel_details(self, channel_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a channel
        
        Args:
            channel_id: YouTube channel ID
        
        Returns:
            Channel details dictionary
        """
        try:
            response = self._call_api(
                self.service.channels().list,
                part='snippet,statistics,contentDetails',
                id=channel_id
            )
            
            if not response.get('items'):
                raise YoutubeAPIError(f"Channel not found: {channel_id}")
            
            channel = response['items'][0]
            snippet = channel.get('snippet', {})
            statistics = channel.get('statistics', {})
            
            return {
                'channel_id': channel_id,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'published_at': snippet.get('publishedAt', ''),
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'subscriber_count': int(statistics.get('subscriberCount', 0)) if statistics.get('subscriberCount') else 0,
                'video_count': int(statistics.get('videoCount', 0)) if statistics.get('videoCount') else 0,
                'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                'country': snippet.get('country', ''),
                'custom_url': snippet.get('customUrl', '')
            }
        
        except Exception as e:
            logger.error(f"Failed to get channel details for {channel_id}: {e}")
            raise YoutubeAPIError(f"Failed to get channel details: {e}")
    
    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent videos from a channel
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch
        
        Returns:
            List of video dictionaries
        """
        try:
            # Search for videos from this channel
            search_response = self._call_api(
                self.service.search().list,
                part='snippet',
                channelId=channel_id,
                maxResults=min(max_results, 50),
                order='date',
                type='video'
            )
            
            videos = []
            for item in search_response.get('items', []):
                if 'videoId' not in item['id']:
                    continue
                
                video_id = item['id']['videoId']
                
                # Get video details
                video_response = self._call_api(
                    self.service.videos().list,
                    part='snippet,statistics,contentDetails',
                    id=video_id
                )
                
                if video_response.get('items'):
                    video = video_response['items'][0]
                    snippet = video.get('snippet', {})
                    statistics = video.get('statistics', {})
                    content_details = video.get('contentDetails', {})
                    
                    videos.append({
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'views': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                        'likes': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else 0,
                        'comments': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else 0,
                        'duration_seconds': iso8601_duration_to_seconds(content_details.get('duration', '')),
                        'url': f'https://www.youtube.com/watch?v={video_id}'
                    })
            
            return videos
        
        except Exception as e:
            logger.error(f"Failed to get channel videos for {channel_id}: {e}")
            raise YoutubeAPIError(f"Failed to get channel videos: {e}")
    
    def get_trending_videos(self, region_code: str = "US", max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Get currently trending videos
        
        Args:
            region_code: Country code (default: US)
            max_results: Maximum number of videos
        
        Returns:
            List of trending videos
        """
        try:
            response = self._call_api(
                self.service.videos().list,
                part='snippet,statistics,contentDetails',
                chart='mostPopular',
                regionCode=region_code,
                maxResults=min(max_results, 50)
            )
            
            videos = []
            for video in response.get('items', []):
                snippet = video.get('snippet', {})
                statistics = video.get('statistics', {})
                content_details = video.get('contentDetails', {})
                
                videos.append({
                    'video_id': video['id'],
                    'title': snippet.get('title', ''),
                    'channel_title': snippet.get('channelTitle', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'views': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                    'likes': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else 0,
                    'comments': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else 0,
                    'duration_seconds': iso8601_duration_to_seconds(content_details.get('duration', '')),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
                    'url': f'https://www.youtube.com/watch?v={video["id"]}'
                })
            
            return videos
        
        except Exception as e:
            logger.error(f"Failed to get trending videos: {e}")
            raise YoutubeAPIError(f"Failed to get trending videos: {e}")
    
    def extract_channel_id_from_url(self, url: str) -> str:
        """
        Extract channel ID from YouTube URL
        
        Args:
            url: YouTube channel URL
        
        Returns:
            Channel ID
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
        
        # If no pattern matches, try to get from custom URL
        try:
            # Try to search for the channel
            search_response = self._call_api(
                self.service.search().list,
                part='snippet',
                q=url,
                type='channel',
                maxResults=1
            )
            
            if search_response.get('items'):
                return search_response['items'][0]['snippet']['channelId']
        
        except Exception:
            pass
        
        raise YoutubeAPIError(f"Could not extract channel ID from URL: {url}")


# For backward compatibility
YouTubeAPIError = YoutubeAPIError
