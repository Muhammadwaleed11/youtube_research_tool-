"""
Filters module for YouTube Research Tool
Contains functions for filtering YouTube video data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger("filters")
logger.setLevel(logging.INFO)


class VideoFilters:
    """
    Class for applying various filters to YouTube video data
    """
    
    def __init__(self):
        self.filters_applied = []
    
    def apply_all_filters(
        self,
        df: pd.DataFrame,
        min_views: Optional[int] = None,
        max_views: Optional[int] = None,
        min_subs: Optional[int] = None,
        max_subs: Optional[int] = None,
        min_vph: Optional[float] = None,
        max_vph: Optional[float] = None,
        min_outlier: Optional[float] = None,
        max_outlier: Optional[float] = None,
        duration_choice: str = "Any",
        published_days: Optional[int] = None,
        include_live: bool = False,
        engagement_min: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Apply all specified filters to the DataFrame
        
        Args:
            df: Input DataFrame with video data
            min_views: Minimum views threshold
            max_views: Maximum views threshold
            min_subs: Minimum channel subscribers
            max_subs: Maximum channel subscribers
            min_vph: Minimum Views Per Hour
            max_vph: Maximum Views Per Hour
            min_outlier: Minimum outlier multiplier
            max_outlier: Maximum outlier multiplier
            duration_choice: Video duration filter
            published_days: Videos published within last N days
            include_live: Include live streams
            engagement_min: Minimum engagement rate (likes+comments)/views
        
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        filtered_df = df.copy()
        self.filters_applied = []
        
        # Apply each filter sequentially
        filtered_df = self.filter_by_views(filtered_df, min_views, max_views)
        filtered_df = self.filter_by_subscribers(filtered_df, min_subs, max_subs)
        filtered_df = self.filter_by_vph(filtered_df, min_vph, max_vph)
        filtered_df = self.filter_by_outlier(filtered_df, min_outlier, max_outlier)
        filtered_df = self.filter_by_duration(filtered_df, duration_choice)
        filtered_df = self.filter_by_publish_date(filtered_df, published_days)
        
        if not include_live:
            filtered_df = self.filter_out_live_streams(filtered_df)
        
        if engagement_min:
            filtered_df = self.filter_by_engagement(filtered_df, engagement_min)
        
        logger.info(f"Applied {len(self.filters_applied)} filters. "
                   f"Results: {len(filtered_df)}/{len(df)} videos")
        
        return filtered_df
    
    def filter_by_views(
        self,
        df: pd.DataFrame,
        min_views: Optional[int] = None,
        max_views: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter videos by view count
        
        Args:
            df: Input DataFrame
            min_views: Minimum views
            max_views: Maximum views
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        if min_views is not None and min_views > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['views'] >= min_views]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Min Views: {min_views:,} (removed {removed})")
        
        if max_views is not None and max_views > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['views'] <= max_views]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Max Views: {max_views:,} (removed {removed})")
        
        return filtered_df
    
    def filter_by_subscribers(
        self,
        df: pd.DataFrame,
        min_subs: Optional[int] = None,
        max_subs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter videos by channel subscriber count
        
        Args:
            df: Input DataFrame
            min_subs: Minimum subscribers
            max_subs: Maximum subscribers
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Handle NaN values in channel_subscribers
        if 'channel_subscribers' in filtered_df.columns:
            # Fill NaN with 0 for filtering
            subs_series = filtered_df['channel_subscribers'].fillna(0)
        else:
            # If column doesn't exist, create dummy series
            subs_series = pd.Series([0] * len(filtered_df))
        
        if min_subs is not None and min_subs > 0:
            original_count = len(filtered_df)
            mask = subs_series >= min_subs
            filtered_df = filtered_df[mask]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Min Subs: {min_subs:,} (removed {removed})")
        
        if max_subs is not None and max_subs > 0:
            original_count = len(filtered_df)
            mask = subs_series <= max_subs
            filtered_df = filtered_df[mask]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Max Subs: {max_subs:,} (removed {removed})")
        
        return filtered_df
    
    def filter_by_vph(
        self,
        df: pd.DataFrame,
        min_vph: Optional[float] = None,
        max_vph: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter videos by Views Per Hour (VPH)
        
        Args:
            df: Input DataFrame
            min_vph: Minimum VPH
            max_vph: Maximum VPH
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Check if VPH column exists, calculate if not
        if 'VPH' not in filtered_df.columns and 'vph' in filtered_df.columns:
            filtered_df['VPH'] = filtered_df['vph']
        elif 'VPH' not in filtered_df.columns:
            # VPH not available, skip filtering
            return filtered_df
        
        if min_vph is not None and min_vph > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['VPH'] >= min_vph]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Min VPH: {min_vph:.1f} (removed {removed})")
        
        if max_vph is not None and max_vph > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['VPH'] <= max_vph]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Max VPH: {max_vph:.1f} (removed {removed})")
        
        return filtered_df
    
    def filter_by_outlier(
        self,
        df: pd.DataFrame,
        min_outlier: Optional[float] = None,
        max_outlier: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter videos by outlier multiplier
        
        Args:
            df: Input DataFrame
            min_outlier: Minimum outlier multiplier
            max_outlier: Maximum outlier multiplier
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Check if outlier column exists
        outlier_col = None
        for col in ['OutlierMultiplier', 'outlier_multiplier', 'multiplier']:
            if col in filtered_df.columns:
                outlier_col = col
                break
        
        if not outlier_col:
            return filtered_df
        
        if min_outlier is not None and min_outlier > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[outlier_col] >= min_outlier]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Min Outlier: {min_outlier}x (removed {removed})")
        
        if max_outlier is not None and max_outlier > 0:
            original_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[outlier_col] <= max_outlier]
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Max Outlier: {max_outlier}x (removed {removed})")
        
        return filtered_df
    
    def filter_by_duration(
        self,
        df: pd.DataFrame,
        duration_choice: str = "Any"
    ) -> pd.DataFrame:
        """
        Filter videos by duration
        
        Args:
            df: Input DataFrame
            duration_choice: Duration category
        
        Returns:
            Filtered DataFrame
        """
        if duration_choice == "Any":
            return df
        
        if 'duration_seconds' not in df.columns:
            return df
        
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        if duration_choice == "Short (<4m)":
            filtered_df = filtered_df[filtered_df['duration_seconds'] < 240]  # 4 minutes = 240 seconds
            duration_label = "Short (<4m)"
        
        elif duration_choice == "Medium (4-20m)":
            filtered_df = filtered_df[
                (filtered_df['duration_seconds'] >= 240) & 
                (filtered_df['duration_seconds'] <= 1200)  # 20 minutes = 1200 seconds
            ]
            duration_label = "Medium (4-20m)"
        
        elif duration_choice == "Long (>20m)":
            filtered_df = filtered_df[filtered_df['duration_seconds'] > 1200]
            duration_label = "Long (>20m)"
        
        else:
            return df
        
        removed = original_count - len(filtered_df)
        if removed > 0:
            self.filters_applied.append(f"Duration: {duration_label} (removed {removed})")
        
        return filtered_df
    
    def filter_by_publish_date(
        self,
        df: pd.DataFrame,
        published_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter videos by publish date
        
        Args:
            df: Input DataFrame
            published_days: Videos published within last N days
        
        Returns:
            Filtered DataFrame
        """
        if published_days is None or published_days <= 0:
            return df
        
        if 'published_at' not in df.columns:
            return df
        
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=published_days)
            
            # Convert published_at strings to datetime
            def parse_date(date_str):
                try:
                    # Remove timezone info for simplicity
                    if 'T' in str(date_str):
                        date_part = str(date_str).split('T')[0]
                        return datetime.strptime(date_part, '%Y-%m-%d')
                    return datetime.strptime(str(date_str), '%Y-%m-%d')
                except:
                    return None
            
            filtered_df['parsed_date'] = filtered_df['published_at'].apply(parse_date)
            
            # Filter by date
            filtered_df = filtered_df[filtered_df['parsed_date'] >= cutoff_date]
            
            # Drop temporary column
            filtered_df = filtered_df.drop('parsed_date', axis=1)
            
            removed = original_count - len(filtered_df)
            if removed > 0:
                self.filters_applied.append(f"Published within {published_days} days (removed {removed})")
        
        except Exception as e:
            logger.error(f"Error filtering by publish date: {e}")
        
        return filtered_df
    
    def filter_out_live_streams(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out live streams from results
        
        Args:
            df: Input DataFrame
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Look for live stream indicators in title or description
        live_keywords = ['live', 'stream', 'premiere', 'premieres', '直播', '스트리밍']
        
        if 'title' in filtered_df.columns:
            mask = ~filtered_df['title'].str.lower().str.contains(
                '|'.join(live_keywords), na=False
            )
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def filter_by_engagement(
        self,
        df: pd.DataFrame,
        engagement_min: float
    ) -> pd.DataFrame:
        """
        Filter videos by engagement rate
        
        Args:
            df: Input DataFrame
            engagement_min: Minimum engagement rate (percentage)
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Check if we have likes and comments data
        has_likes = 'likes' in filtered_df.columns
        has_comments = 'comments' in filtered_df.columns
        
        if not (has_likes and has_comments):
            return filtered_df
        
        # Calculate engagement rate
        filtered_df['engagement_rate'] = (
            (filtered_df['likes'].fillna(0) + filtered_df['comments'].fillna(0)) / 
            filtered_df['views'].replace(0, 1) * 100
        )
        
        original_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['engagement_rate'] >= engagement_min]
        
        removed = original_count - len(filtered_df)
        if removed > 0:
            self.filters_applied.append(f"Min Engagement: {engagement_min}% (removed {removed})")
        
        return filtered_df
    
    def filter_by_channel_size(
        self,
        df: pd.DataFrame,
        channel_size: str = "Any"
    ) -> pd.DataFrame:
        """
        Filter by channel size category
        
        Args:
            df: Input DataFrame
            channel_size: Size category
        
        Returns:
            Filtered DataFrame
        """
        if channel_size == "Any":
            return df
        
        if 'channel_subscribers' not in df.columns:
            return df
        
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        # Fill NaN with 0 for filtering
        subs_series = filtered_df['channel_subscribers'].fillna(0)
        
        if channel_size == "Micro (<1K)":
            mask = subs_series < 1000
        elif channel_size == "Small (1K-10K)":
            mask = (subs_series >= 1000) & (subs_series < 10000)
        elif channel_size == "Medium (10K-100K)":
            mask = (subs_series >= 10000) & (subs_series < 100000)
        elif channel_size == "Large (100K-1M)":
            mask = (subs_series >= 100000) & (subs_series < 1000000)
        elif channel_size == "Mega (>1M)":
            mask = subs_series >= 1000000
        else:
            return df
        
        filtered_df = filtered_df[mask]
        removed = original_count - len(filtered_df)
        if removed > 0:
            self.filters_applied.append(f"Channel Size: {channel_size} (removed {removed})")
        
        return filtered_df
    
    def get_filter_summary(self) -> List[str]:
        """
        Get summary of applied filters
        
        Returns:
            List of filter descriptions
        """
        return self.filters_applied.copy()


# Legacy function for backward compatibility
def apply_filters_to_df(
    df: pd.DataFrame,
    min_subscribers: Optional[int] = None,
    max_subscribers: Optional[int] = None,
    min_views: Optional[int] = None,
    max_views: Optional[int] = None,
    duration_choice: str = "Any",
    **kwargs
) -> pd.DataFrame:
    """
    Legacy function for applying filters (maintains compatibility)
    
    Args:
        df: Input DataFrame
        min_subscribers: Minimum channel subscribers
        max_subscribers: Maximum channel subscribers
        min_views: Minimum views
        max_views: Maximum views
        duration_choice: Duration category
        **kwargs: Additional filter parameters
    
    Returns:
        Filtered DataFrame
    """
    filter_instance = VideoFilters()
    
    # Map legacy parameters to new ones
    filters = {
        'min_subs': min_subscribers,
        'max_subs': max_subscribers,
        'min_views': min_views,
        'max_views': max_views,
        'duration_choice': duration_choice
    }
    
    # Add any additional filters from kwargs
    filters.update(kwargs)
    
    return filter_instance.apply_all_filters(df, **filters)


# Helper function for common filter combinations
def get_preset_filters(preset_name: str) -> Dict[str, Any]:
    """
    Get predefined filter presets
    
    Args:
        preset_name: Name of preset
    
    Returns:
        Dictionary of filter parameters
    """
    presets = {
        'viral_finder': {
            'min_vph': 100,
            'min_outlier': 3.0,
            'published_days': 7,
            'duration_choice': 'Any'
        },
        'growing_channels': {
            'min_subs': 1000,
            'max_subs': 10000,
            'min_vph': 10,
            'published_days': 30
        },
        'high_engagement': {
            'engagement_min': 5.0,
            'min_views': 1000,
            'published_days': 30
        },
        'short_form': {
            'duration_choice': 'Short (<4m)',
            'min_vph': 50,
            'published_days': 7
        },
        'long_form': {
            'duration_choice': 'Long (>20m)',
            'min_views': 5000,
            'published_days': 30
        }
    }
    
    return presets.get(preset_name, {})
