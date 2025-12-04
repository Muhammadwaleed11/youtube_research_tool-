
Competitor Tracker module for YouTube Research Tool
Handles tracking, monitoring, and analyzing competitor channels
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger("competitor_tracker")
logger.setLevel(logging.INFO)


class CompetitorTracker:
    """
    Main competitor tracking class
    Tracks YouTube channels and monitors their performance
    """
    
    def __init__(self, db, youtube_api):
        """
        Initialize competitor tracker
        
        Args:
            db: Database instance
            youtube_api: YouTube API instance
        """
        self.db = db
        self.youtube = youtube_api
        self.last_update_time = {}
        logger.info("Competitor Tracker initialized")
    
    def add_competitor(self, channel_url: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Add a new competitor channel to track
        
        Args:
            channel_url: YouTube channel URL
            user_id: User identifier
        
        Returns:
            Dictionary with success status and channel info
        """
        try:
            logger.info(f"Adding competitor: {channel_url}")
            
            # Extract channel ID from URL
            channel_id = self.youtube.extract_channel_id_from_url(channel_url)
            if not channel_id:
                return {
                    'success': False,
                    'message': 'Could not extract channel ID from URL',
                    'channel_id': None
                }
            
            # Check if already exists
            existing = self.db.get_competitor_by_channel_id(channel_id)
            if existing:
                return {
                    'success': True,
                    'message': 'Competitor already exists',
                    'channel_id': channel_id,
                    'channel_info': existing
                }
            
            # Get channel details from YouTube
            channel_details = self.youtube.get_channel_details(channel_id)
            if not channel_details:
                return {
                    'success': False,
                    'message': 'Could not fetch channel details from YouTube',
                    'channel_id': channel_id
                }
            
            # Get recent videos to calculate average views
            recent_videos = self.youtube.get_channel_videos(channel_id, max_results=20)
            avg_views = self._calculate_average_views(recent_videos)
            
            # Prepare competitor data
            competitor_data = {
                'user_id': user_id,
                'channel_id': channel_id,
                'channel_name': channel_details.get('title', 'Unknown Channel'),
                'channel_url': f"https://www.youtube.com/channel/{channel_id}",
                'description': channel_details.get('description', '')[:500],
                'subscriber_count': channel_details.get('subscriber_count', 0),
                'video_count': channel_details.get('video_count', 0),
                'view_count': channel_details.get('view_count', 0),
                'average_views': avg_views,
                'country': channel_details.get('country', ''),
                'thumbnail_url': channel_details.get('thumbnail_url', ''),
                'added_date': datetime.now().isoformat(),
                'last_checked': datetime.now().isoformat(),
                'tracking_active': True
            }
            
            # Save to database
            success = self.db.add_competitor(**competitor_data)
            
            if success:
                # Save initial update
                self.db.save_competitor_update(
                    competitor_id=competitor_data.get('id', 1),
                    data={
                        'subscriber_count': competitor_data['subscriber_count'],
                        'video_count': competitor_data['video_count'],
                        'view_count': competitor_data['view_count'],
                        'average_views': competitor_data['average_views'],
                        'new_videos': 0
                    }
                )
                
                logger.info(f"Competitor added successfully: {competitor_data['channel_name']}")
                return {
                    'success': True,
                    'message': 'Competitor added successfully',
                    'channel_id': channel_id,
                    'channel_info': competitor_data
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save competitor to database',
                    'channel_id': channel_id
                }
        
        except Exception as e:
            logger.error(f"Error adding competitor {channel_url}: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'channel_id': None
            }
    
    def _calculate_average_views(self, videos: List[Dict[str, Any]]) -> float:
        """
        Calculate average views from recent videos
        
        Args:
            videos: List of video dictionaries
        
        Returns:
            Average views
        """
        if not videos:
            return 0.0
        
        total_views = sum(v.get('views', 0) for v in videos)
        return total_views / len(videos)
    
    def get_competitor_updates(self, channel_id: str) -> Dict[str, Any]:
        """
        Check for updates on a competitor channel
        
        Args:
            channel_id: YouTube channel ID
        
        Returns:
            Dictionary with update information
        """
        try:
            logger.info(f"Checking updates for channel: {channel_id}")
            
            # Get current channel info from database
            competitor = self.db.get_competitor_by_channel_id(channel_id)
            if not competitor:
                return {'error': 'Competitor not found'}
            
            # Get fresh data from YouTube
            channel_details = self.youtube.get_channel_details(channel_id)
            if not channel_details:
                return {'error': 'Could not fetch channel details'}
            
            # Get recent videos
            recent_videos = self.youtube.get_channel_videos(channel_id, max_results=50)
            
            # Calculate metrics
            current_subs = channel_details.get('subscriber_count', 0)
            current_videos = channel_details.get('video_count', 0)
            current_views = channel_details.get('view_count', 0)
            avg_views = self._calculate_average_views(recent_videos)
            
            # Compare with previous data
            prev_subs = competitor.get('subscriber_count', 0)
            prev_videos = competitor.get('video_count', 0)
            
            sub_change = current_subs - prev_subs
            video_change = current_videos - prev_videos
            
            # Identify viral videos (high VPH)
            viral_videos = []
            for video in recent_videos:
                vph = self._calculate_vph_for_video(video)
                if vph > 1000:  # Threshold for viral
                    viral_videos.append({
                        'title': video.get('title', 'Unknown'),
                        'views': video.get('views', 0),
                        'vph': vph,
                        'published_at': video.get('published_at'),
                        'url': video.get('url')
                    })
            
            # Prepare update data
            update_data = {
                'subscriber_count': current_subs,
                'video_count': current_videos,
                'view_count': current_views,
                'average_views': avg_views,
                'new_videos': max(video_change, 0),  # Only positive changes
                'subscriber_change': sub_change,
                'viral_videos': viral_videos,
                'last_checked': datetime.now().isoformat()
            }
            
            # Update database
            self.db.update_competitor(channel_id, **update_data)
            
            # Check for alerts
            alerts = self._check_for_alerts(competitor, update_data)
            
            logger.info(f"Updates found for {competitor.get('channel_name')}: "
                       f"+{update_data['new_videos']} videos, "
                       f"+{update_data['subscriber_change']} subs")
            
            return {
                'success': True,
                'channel_name': competitor.get('channel_name'),
                'updates': update_data,
                'alerts': alerts
            }
        
        except Exception as e:
            logger.error(f"Error getting updates for {channel_id}: {e}")
            return {'error': str(e)}
    
    def _calculate_vph_for_video(self, video: Dict[str, Any]) -> float:
        """
        Calculate VPH for a video
        
        Args:
            video: Video dictionary
        
        Returns:
            VPH value
        """
        try:
            from modules.utils import calculate_vph
            return calculate_vph(
                video.get('views', 0),
                video.get('published_at', '')
            )
        except:
            return 0.0
    
    def _check_for_alerts(self, competitor: Dict[str, Any], updates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if updates trigger any alerts
        
        Args:
            competitor: Competitor information
            updates: Update data
        
        Returns:
            List of alerts
        """
        alerts = []
        channel_name = competitor.get('channel_name', 'Unknown')
        
        # Check for new videos
        if updates.get('new_videos', 0) > 0:
            alerts.append({
                'type': 'new_video',
                'message': f"{channel_name} uploaded {updates['new_videos']} new video(s)",
                'priority': 'info'
            })
        
        # Check for subscriber spike
        sub_change = updates.get('subscriber_change', 0)
        if sub_change > 1000:
            alerts.append({
                'type': 'subscriber_spike',
                'message': f"{channel_name} gained {sub_change:,} new subscribers",
                'priority': 'warning'
            })
        
        # Check for viral videos
        viral_videos = updates.get('viral_videos', [])
        if viral_videos:
            top_video = max(viral_videos, key=lambda x: x.get('vph', 0))
            alerts.append({
                'type': 'viral_video',
                'message': f"{channel_name} has a viral video: {top_video['title']} (VPH: {top_video['vph']:.0f})",
                'priority': 'critical'
            })
        
        # Check for significant growth in average views
        prev_avg = competitor.get('average_views', 0)
        current_avg = updates.get('average_views', 0)
        if prev_avg > 0 and current_avg > prev_avg * 1.5:  # 50% increase
            alerts.append({
                'type': 'view_growth',
                'message': f"{channel_name}'s average views increased by {((current_avg/prev_avg)-1)*100:.0f}%",
                'priority': 'info'
            })
        
        # Save alerts to database
        for alert in alerts:
            self.db.add_alert(
                user_id=competitor.get('user_id', 'default'),
                competitor_id=competitor.get('id'),
                alert_type=alert['type'],
                alert_message=alert['message'],
                alert_data=json.dumps(alert) if 'json' in locals() else str(alert)
            )
        
        return alerts
    
    def send_alerts(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Check all tracked competitors for alerts
        
        Args:
            user_id: User identifier
        
        Returns:
            List of alerts
        """
        try:
            logger.info(f"Checking alerts for user: {user_id}")
            
            # Get all tracked competitors
            competitors = self.db.get_all_competitors(user_id)
            
            all_alerts = []
            for competitor in competitors:
                # Check if we should update this competitor
                channel_id = competitor.get('channel_id')
                last_checked = competitor.get('last_checked')
                
                # Update if never checked or checked more than 6 hours ago
                if not last_checked or self._should_update(last_checked):
                    updates = self.get_competitor_updates(channel_id)
                    if 'alerts' in updates:
                        all_alerts.extend(updates['alerts'])
            
            # Get unread alerts from database
            db_alerts = self.db.get_recent_alerts(user_id, limit=20)
            
            # Mark alerts as read
            for alert in db_alerts:
                if not alert.get('is_read'):
                    self.db.mark_alert_read(alert['id'])
            
            # Combine alerts
            combined_alerts = all_alerts + [
                {
                    'type': a.get('alert_type'),
                    'message': a.get('alert_message'),
                    'priority': 'info',
                    'timestamp': a.get('created_at')
                }
                for a in db_alerts
]
logger.info(f"Found {len(combined_alerts)} alerts")
            return combined_alerts
        
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
            return []
    
    def _should_update(self, last_checked: str) -> bool:
        """
        Check if competitor should be updated
        
        Args:
            last_checked: Last check timestamp
        
        Returns:
            True if should update
        """
        try:
            last_time = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
            current_time = datetime.now(last_time.tzinfo) if last_time.tzinfo else datetime.utcnow()
            
            hours_since_update = (current_time - last_time).total_seconds() / 3600
            return hours_since_update >= 6  # Update every 6 hours
        
        except Exception:
            return True  # Update if can't parse timestamp
    
    def get_all_competitors(self, user_id: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all tracked competitors
        
        Args:
            user_id: User identifier
            limit: Maximum number to return
        
        Returns:
            List of competitor dictionaries
        """
        return self.db.get_all_competitors(user_id, limit)
    
    def get_competitor_by_name(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get competitor by channel name
        
        Args:
            channel_name: Channel name
        
        Returns:
            Competitor dictionary or None
        """
        competitors = self.get_all_competitors()
        for comp in competitors:
            if comp.get('channel_name') == channel_name:
                return comp
        return None
    
    def get_competitor_metrics(self, channel_id: str) -> Dict[str, Any]:
        """
        Get metrics for a competitor
        
        Args:
            channel_id: YouTube channel ID
        
        Returns:
            Dictionary of metrics
        """
        competitor = self.db.get_competitor_by_channel_id(channel_id)
        if not competitor:
            return {}
        
        # Get historical updates
        updates_df = self.db.get_competitor_updates(competitor.get('id'), days=30)
        
        metrics = {
            'basic_info': {
                'channel_name': competitor.get('channel_name'),
                'subscriber_count': competitor.get('subscriber_count'),
                'video_count': competitor.get('video_count'),
                'average_views': competitor.get('average_views'),
                'last_checked': competitor.get('last_checked')
            },
            'growth_metrics': self._calculate_growth_metrics(updates_df),
            'recent_updates': updates_df.to_dict('records') if not updates_df.empty else []
        }
        
        return metrics
    
    def _calculate_growth_metrics(self, updates_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate growth metrics from updates
        
        Args:
            updates_df: DataFrame of historical updates
        
        Returns:
            Dictionary of growth metrics
        """
        if updates_df.empty:
            return {}
        
        try:
            # Calculate daily subscriber growth
            if 'subscriber_count' in updates_df.columns:
                latest_subs = updates_df['subscriber_count'].iloc[-1]
                oldest_subs = updates_df['subscriber_count'].iloc[0]
                days_diff = len(updates_df)
                
                if days_diff > 0 and oldest_subs > 0:
                    daily_growth = (latest_subs - oldest_subs) / days_diff
                    growth_rate = ((latest_subs / oldest_subs) ** (365/days_diff) - 1) * 100
                else:
                    daily_growth = 0
                    growth_rate = 0
            else:
                daily_growth = 0
                growth_rate = 0
            
            # Calculate video upload frequency
            if 'new_videos' in updates_df.columns:
                total_new_videos = updates_df['new_videos'].sum()
                days_with_updates = len(updates_df[updates_df['new_videos'] > 0])
                
                if days_with_updates > 0:
                    avg_videos_per_day = total_new_videos / days_with_updates
                else:
                    avg_videos_per_day = 0
            else:
                avg_videos_per_day = 0
            
            return {
                'daily_subscriber_growth': daily_growth,
                'annual_growth_rate_percent': growth_rate,
                'average_videos_per_day': avg_videos_per_day,
                'total_updates_tracked': len(updates_df)
            }
        
        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            return {}
    
    def get_comparison_data(self, your_channel_id: str, competitor_id: str) -> Dict[str, Any]:
        """
        Compare your channel with a competitor
        
        Args:
            your_channel_id: Your channel ID
            competitor_id: Competitor channel ID
        
        Returns:
            Comparison data
        """
        try:
            # Get your channel details
            your_channel = self.youtube.get_channel_details(your_channel_id)
            your_videos = self.youtube.get_channel_videos(your_channel_id, max_results=20)
            your_avg_views = self._calculate_average_views(your_videos)
            
            # Get competitor details
            competitor = self.db.get_competitor_by_channel_id(competitor_id)
            if not competitor:
                # Try to get from YouTube if not in database
                competitor_details = self.youtube.get_channel_details(competitor_id)
                if competitor_details:
                    comp_videos = self.youtube.get_channel_videos(competitor_id, max_results=20)
                    comp_avg_views = self._calculate_average_views(comp_videos)
                    competitor = {
                        'subscriber_count': competitor_details.get('subscriber_count', 0),
                        'video_count': competitor_details.get('video_count', 0),
                        'average_views': comp_avg_views,
                        'channel_name': competitor_details.get('title', 'Unknown')
                    }
                else:
                    competitor = {
                        'subscriber_count': 0,
                        'video_count': 0,
                        'average_views': 0,
                        'channel_name': 'Unknown'
                    }
            
            # Calculate comparison metrics
            comparison = {
                'metrics': ['Subscribers', 'Total Videos', 'Average Views'],
                'your_channel': [
                    your_channel.get('subscriber_count', 0),
                    your_channel.get('video_count', 0),
                    your_avg_views
                ],
                'competitor': [
                    competitor.get('subscriber_count', 0),
                    competitor.get('video_count', 0),
                    competitor.get('average_views', 0)
                ],
                'ratio': [
                    your_channel.get('subscriber_count', 1) / max(competitor.get('subscriber_count', 1), 1),
                    your_channel.get('video_count', 1) / max(competitor.get('video_count', 1), 1),
                    your_avg_views / max(competitor.get('average_views', 1), 1)
                ]
            }
            
            # Add insights
            insights = []
            
            sub_ratio = comparison['ratio'][0]
            if sub_ratio < 0.5:
                insights.append("Your channel has less than half the subscribers of the competitor")
            elif sub_ratio > 2:
                insights.append("Your channel has more than double the subscribers of the competitor")
            
            view_ratio = comparison['ratio'][2]
            if view_ratio < 0.5:
                insights.append("Your videos get less than half the average views of competitor's videos")
            elif view_ratio > 2:
                insights.append("Your videos get more than double the average views of competitor's videos")
            
            comparison['insights'] = insights
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error comparing channels: {e}")
            return {'error': str(e)}
    
    def get_trending_competitors(self, niche: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find trending competitors in a specific niche
        
        Args:
            niche: Content niche
            limit: Maximum number to return
        
        Returns:
            List of trending competitors
        """
        # This is a simplified version - in production, you'd want more sophisticated logic
        competitors = self.get_all_competitors(limit=100)
        
        if niche:
            # Filter by niche (simple keyword matching)
            niche_keywords = niche.lower().split()
            filtered = []
            
            for comp in competitors:
                description = comp.get('description', '').lower()
                channel_name = comp.get('channel_name', '').lower()
                
                # Check if any niche keyword appears in description or name
                if any(keyword in description or keyword in channel_name 
                      for keyword in niche_keywords):
                    filtered.append(comp)
            
            competitors = filtered
        
        # Sort by growth metrics (simplified - sort by subscribers)
        competitors.sort(key=lambda x: x.get('subscriber_count', 0), reverse=True)
        
        return competitors[:limit]
    
    def get_viral_videos(self, channel_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get viral videos from a competitor
        
        Args:
            channel_id: YouTube channel ID
            days: Look back days
        
        Returns:
            List of viral videos
        """
        try:
            # Get recent videos
            recent_videos = self.youtube.get_channel_videos(channel_id, max_results=50)
            
            viral_videos = []
            for video in recent_videos:
                vph = self._calculate_vph_for_video(video)
                if vph > 500:  # Threshold for "viral"
                    viral_videos.append({
                        'title': video.get('title', 'Unknown'),
                        'views': video.get('views', 0),
                        'vph': vph,
                        'published_at': video.get('published_at'),
                        'url': video.get('url'),
                        'engagement_rate': self._calculate_engagement_rate(video)
                    })
            # Sort by VPH (highest first)
            viral_videos.sort(key=lambda x: x.get('vph', 0), reverse=True)
            
            return viral_videos[:10]  # Return top 10
        
        except Exception as e:
            logger.error(f"Error getting viral videos for {channel_id}: {e}")
            return []
    
    def _calculate_engagement_rate(self, video: Dict[str, Any]) -> float:
        """
        Calculate engagement rate for a video
        
        Args:
            video: Video dictionary
        
        Returns:
            Engagement rate percentage
        """
        try:
            likes = video.get('likes', 0)
            comments = video.get('comments', 0)
            views = video.get('views', 1)  # Avoid division by zero
            
            return ((likes + comments) / views) * 100
        except:
            return 0.0
    
    def get_export_data(self, channel_id: str) -> pd.DataFrame:
        """
        Get competitor data for export
        
        Args:
            channel_id: YouTube channel ID
        
        Returns:
            DataFrame with export data
        """
        competitor = self.db.get_competitor_by_channel_id(channel_id)
        if not competitor:
            return pd.DataFrame()
        
        # Get historical updates
        updates_df = self.db.get_competitor_updates(competitor.get('id'), days=90)
        
        if updates_df.empty:
            # Create minimal DataFrame
            data = {
                'channel_name': [competitor.get('channel_name')],
                'subscribers': [competitor.get('subscriber_count')],
                'videos': [competitor.get('video_count')],
                'average_views': [competitor.get('average_views')],
                'last_updated': [competitor.get('last_checked')]
            }
            return pd.DataFrame(data)
        
        # Prepare export DataFrame
        export_df = updates_df.copy()
        export_df['channel_name'] = competitor.get('channel_name')
        
        # Reorder columns
        column_order = ['check_date', 'channel_name', 'subscriber_count', 
                       'video_count', 'view_count', 'average_views', 'new_videos']
        export_df = export_df[[col for col in column_order if col in export_df.columns]]
        
        return export_df
    
    def remove_competitor(self, channel_id: str, user_id: str = "default") -> bool:
        """
        Remove a competitor from tracking
        
        Args:
            channel_id: YouTube channel ID
            user_id: User identifier
        
        Returns:
            True if successful
        """
        try:
            # In a real implementation, you would mark as inactive or delete from database
            # For now, we'll just update tracking_active to False
            competitor = self.db.get_competitor_by_channel_id(channel_id)
            if competitor:
                self.db.update_competitor(channel_id, tracking_active=False)
                logger.info(f"Removed competitor: {channel_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error removing competitor {channel_id}: {e}")
            return False


# Helper function for JSON serialization (if needed)
try:
    import json
except ImportError:
    import simplejson as json
```
