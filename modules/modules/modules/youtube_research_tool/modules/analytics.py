Analytics module for YouTube Research Tool
Provides analytics, insights, and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

# Configure logging
logger = logging.getLogger("analytics")
logger.setLevel(logging.INFO)


class AnalyticsService:
    """
    Analytics service for generating insights and performance metrics
    """
    
    def __init__(self, db):
        """
        Initialize analytics service
        
        Args:
            db: Database instance
        """
        self.db = db
        logger.info("Analytics Service initialized")
    
    # ==================== BASIC METRICS ====================
    
    def get_overall_metrics(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Get overall analytics metrics
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            # Get database stats
            db_stats = self.db.get_stats()
            metrics.update(db_stats)
            
            # Calculate additional metrics
            metrics['analysis_date'] = datetime.now().isoformat()
            metrics['user_id'] = user_id
            
            # Calculate average VPH from recent searches
            avg_vph = self._calculate_average_vph(user_id)
            metrics['average_vph'] = avg_vph
            
            # Calculate success rate (videos with VPH > 100)
            success_rate = self._calculate_success_rate(user_id)
            metrics['success_rate_percent'] = success_rate
            
            # Get top niche
            top_niche = self._get_top_niche(user_id)
            metrics['top_niche'] = top_niche
            
            # Calculate engagement rate
            engagement_rate = self._calculate_average_engagement(user_id)
            metrics['average_engagement_rate'] = engagement_rate
            
            # Add performance labels
            metrics['performance_label'] = self._get_performance_label(avg_vph, success_rate)
            
            logger.info(f"Generated overall metrics for user {user_id}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting overall metrics: {e}")
            return {
                'error': str(e),
                'analysis_date': datetime.now().isoformat()
            }
    
    def _calculate_average_vph(self, user_id: str) -> float:
        """
        Calculate average VPH from recent searches
        
        Args:
            user_id: User identifier
        
        Returns:
            Average VPH
        """
        try:
            # Get recent searches
            searches = self.db.get_search_history(limit=50)
            if not searches:
                return 0.0
            
            total_vph = 0
            count = 0
            
            for search in searches:
                search_id = search.get('id')
                results = self.db.get_results_for_search(search_id)
                
                if results is not None and not results.empty:
                    if 'vph' in results.columns:
                        avg_search_vph = results['vph'].mean()
                        total_vph += avg_search_vph
                        count += 1
            
            return total_vph / count if count > 0 else 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_success_rate(self, user_id: str) -> float:
        """
        Calculate success rate (percentage of videos with VPH > 100)
        
        Args:
            user_id: User identifier
        
        Returns:
            Success rate percentage
        """
        try:
            searches = self.db.get_search_history(limit=50)
            if not searches:
                return 0.0
            
            total_videos = 0
            successful_videos = 0
            
            for search in searches:
                search_id = search.get('id')
                results = self.db.get_results_for_search(search_id)
                
                if results is not None and not results.empty:
                    if 'vph' in results.columns:
                        total_videos += len(results)
                        successful_videos += len(results[results['vph'] > 100])
            
            if total_videos == 0:
                return 0.0
            
            return (successful_videos / total_videos) * 100
        
        except Exception:
            return 0.0
    
    def _get_top_niche(self, user_id: str) -> str:
        """
        Identify top performing niche
        
        Args:
            user_id: User identifier
        
        Returns:
            Top niche name
        """
        try:
            searches = self.db.get_search_history(limit=100)
            if not searches:
                return "General"
            
            # Simple keyword analysis
            keywords = []
            for search in searches:
                query = search.get('query', '').lower()
                keywords.extend(query.split())
            
            # Common niches and their keywords
            niches = {
                'Technology': ['tech', 'programming', 'coding', 'software', 'computer', 'python', 'java'],
                'Gaming': ['game', 'gaming', 'playthrough', 'walkthrough', 'esports'],
                'Education': ['learn', 'tutorial', 'howto', 'education', 'course', 'lesson'],
                'Entertainment': ['funny', 'comedy', 'entertainment', 'vlog', 'prank'],
                'Business': ['business', 'marketing', 'entrepreneur', 'startup', 'money'],
                'Health': ['fitness', 'health', 'workout', 'diet', 'nutrition', 'yoga'],
                'Music': ['music', 'song', 'cover', 'instrument', 'guitar', 'piano']
            }
            
            # Count keyword matches
            niche_scores = {niche: 0 for niche in niches.keys()}
            
            for keyword in keywords:
                for niche, niche_keywords in niches.items():
                    if keyword in niche_keywords:
                        niche_scores[niche] += 1
            
            # Get top niche
            if niche_scores:
                top_niche = max(niche_scores, key=niche_scores.get)
                if niche_scores[top_niche] > 0:
                    return top_niche
            
            return "General"
        
        except Exception:
            return "General"
    
    def _calculate_average_engagement(self, user_id: str) -> float:
        """
        Calculate average engagement rate
        
        Args:
            user_id: User identifier
        
        Returns:
            Average engagement rate percentage
        """
        try:
            searches = self.db.get_search_history(limit=50)
            if not searches:
                return 0.0
            
            total_engagement = 0
            count = 0
            
            for search in searches:
                search_id = search.get('id')
                results = self.db.get_results_for_search(search_id)
                
                if results is not None and not results.empty:
                    if 'likes' in results.columns and 'comments' in results.columns and 'views' in results.columns:
                        results['engagement'] = (
                            (results['likes'].fillna(0) + results['comments'].fillna(0)) / 
                            results['views'].replace(0, 1) * 100
                        )
                        
                        avg_engagement = results['engagement'].mean()
                        if not pd.isna(avg_engagement):
                            total_engagement += avg_engagement
                            count += 1
            
            return total_engagement / count if count > 0 else 0.0
        
        except Exception:
            return 0.0
    
    def _get_performance_label(self, avg_vph: float, success_rate: float) -> str:
        """
        Get performance label based on metrics
        
        Args:
            avg_vph: Average VPH
            success_rate: Success rate percentage
        
        Returns:
            Performance label
        """
        if avg_vph > 500 and success_rate > 30:
            return "Excellent"
        elif avg_vph > 200 and success_rate > 15:
            return "Good"
        elif avg_vph > 50 and success_rate > 5:
            return "Average"
        else:
            return "Needs Improvement"
    
    # ==================== TREND ANALYSIS ====================
    
    def get_trends(self, user_id: str = "default", days: int = 30) -> pd.DataFrame:
        """
        Get trend data over time
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
        
        Returns:
            DataFrame with trend data
        """
        try:
            # Get searches within timeframe
            searches = self.db.get_search_history(limit=1000)
            if not searches:
                return self._create_empty_trends_df(days)
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_searches = []
            
            for search in searches:
                try:
                    search_date = datetime.fromisoformat(search.get('created_at').replace('Z', '+00:00'))
                    if search_date >= cutoff_date:
                        recent_searches.append(search)
                except:
                    continue
            
            if not recent_searches:
                return self._create_empty_trends_df(days)
            
            # Group by date and calculate metrics
            trends_data = []
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).date()
                
                # Get searches for this date
                date_searches = []
                for search in recent_searches:
                    try:
                        search_date = datetime.fromisoformat(
                            search.get('created_at').replace('Z', '+00:00')
                        ).date()
                        if search_date == date:
                            date_searches.append(search)
                    except:
                        continue
                
                if not date_searches:
                    trends_data.append({
                        'date': date,
                        'searches': 0,
                        'avg_vph': 0,
                        'success_rate': 0,
                        'engagement_rate': 0
                    })
                    continue
                
                # Calculate metrics for this date
                daily_vph = []
                daily_success = []
                daily_engagement = []
                
                for search in date_searches:
                    search_id = search.get('id')
                    results = self.db.get_results_for_search(search_id)
                    
                    if results is not None and not results.empty:
                        # VPH
                        if 'vph' in results.columns:
                            avg_vph = results['vph'].mean()
                            if not pd.isna(avg_vph):
                                daily_vph.append(avg_vph)
                            
                            # Success rate (VPH > 100)
                            success_count = len(results[results['vph'] > 100])
                            if len(results) > 0:
                                daily_success.append((success_count / len(results)) * 100)
                        
                        # Engagement
                        if 'likes' in results.columns and 'comments' in results.columns and 'views' in results.columns:
                            results['engagement'] = (
                                (results['likes'].fillna(0) + results['comments'].fillna(0)) / 
                                results['views'].replace(0, 1) * 100
                            )
                            avg_engagement = results['engagement'].mean()
                            if not pd.isna(avg_engagement):
                                daily_engagement.append(avg_engagement)
                
                trends_data.append({
                    'date': date,
                    'searches': len(date_searches),
                    'avg_vph': np.mean(daily_vph) if daily_vph else 0,
                    'success_rate': np.mean(daily_success) if daily_success else 0,
                    'engagement_rate': np.mean(daily_engagement) if daily_engagement else 0
                })
                 # Create DataFrame and sort by date
            df = pd.DataFrame(trends_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Generated trends for {days} days")
            return df
        
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return self._create_empty_trends_df(days)
    
    def _create_empty_trends_df(self, days: int) -> pd.DataFrame:
        """Create empty trends DataFrame"""
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(days)]
        
        data = {
            'date': dates,
            'searches': [0] * days,
            'avg_vph': [0] * days,
            'success_rate': [0] * days,
            'engagement_rate': [0] * days
        }
        
        return pd.DataFrame(data)
    
    # ==================== CONTENT ANALYSIS ====================
    
    def get_top_formats(self, user_id: str = "default") -> pd.DataFrame:
        """
        Analyze top performing content formats
        
        Args:
            user_id: User identifier
        
        Returns:
            DataFrame with format analysis
        """
        try:
            searches = self.db.get_search_history(limit=100)
            if not searches:
                return self._create_default_formats_df()
            
            # Define format categories
            formats_data = {
                'Short (<4m)': {'count': 0, 'total_vph': 0, 'success_count': 0},
                'Medium (4-20m)': {'count': 0, 'total_vph': 0, 'success_count': 0},
                'Long (>20m)': {'count': 0, 'total_vph': 0, 'success_count': 0}
            }
            
            for search in searches:
                search_id = search.get('id')
                results = self.db.get_results_for_search(search_id)
                
                if results is not None and not results.empty:
                    if 'duration_seconds' in results.columns and 'vph' in results.columns:
                        # Categorize each video
                        for _, row in results.iterrows():
                            duration = row.get('duration_seconds', 0)
                            vph = row.get('vph', 0)
                            
                            if duration < 240:  # < 4 minutes
                                format_name = 'Short (<4m)'
                            elif duration <= 1200:  # 4-20 minutes
                                format_name = 'Medium (4-20m)'
                            else:  # > 20 minutes
                                format_name = 'Long (>20m)'
                            
                            formats_data[format_name]['count'] += 1
                            formats_data[format_name]['total_vph'] += vph
                            if vph > 100:
                                formats_data[format_name]['success_count'] += 1
            
            # Calculate metrics
            format_rows = []
            for format_name, data in formats_data.items():
                count = data['count']
                if count > 0:
                    avg_vph = data['total_vph'] / count
                    success_rate = (data['success_count'] / count) * 100
                else:
                    avg_vph = 0
                    success_rate = 0
                
                format_rows.append({
                    'format': format_name,
                    'video_count': count,
                    'average_vph': round(avg_vph, 2),
                    'success_rate_percent': round(success_rate, 2)
                })
            
            df = pd.DataFrame(format_rows)
            
            # Add recommendations
            if not df.empty:
                best_format = df.loc[df['average_vph'].idxmax()]
                df['recommendation'] = df.apply(
                    lambda row: '⭐ Best Performing' if row['format'] == best_format['format'] else '',
                    axis=1
                )
            
            logger.info("Generated format analysis")
            return df
        
        except Exception as e:
            logger.error(f"Error analyzing formats: {e}")
            return self._create_default_formats_df()
    
    def _create_default_formats_df(self) -> pd.DataFrame:
        """Create default formats DataFrame"""
        data = {
            'format': ['Short (<4m)', 'Medium (4-20m)', 'Long (>20m)'],
            'video_count': [0, 0, 0],
            'average_vph': [0, 0, 0],
            'success_rate_percent': [0, 0, 0],
            'recommendation': ['', '', '']
        }
        return pd.DataFrame(data)
    
    def get_video_engagement(self, user_id: str = "default") -> pd.DataFrame:
        """
        Get engagement analysis for videos
        
        Args:
            user_id: User identifier
        
        Returns:
            DataFrame with engagement data
        """
        try:
            searches = self.db.get_search_history(limit=50)
            if not searches:
                return pd.DataFrame()
            
            engagement_data = []
            
            for search in searches:
                search_id = search.get('id')
                results = self.db.get_results_for_search(search_id)
                
                if results is not None and not results.empty:
                    if 'likes' in results.columns and 'comments' in results.columns and 'views' in results.columns:
                        for _, row in results.iterrows():
                            title = str(row.get('title', ''))[:50]  # Truncate long titles
                            views = row.get('views', 0)
                            likes = row.get('likes', 0)
                            comments = row.get('comments', 0)
                            
                            if views > 0:
                                engagement_rate = ((likes + comments) / views) * 100
                            else:
                                engagement_rate = 0
                            
                            engagement_data.append({
                                'video_title': title,
                                'views': views,
                                'likes': likes,
                                'comments': comments,
                                'engagement_rate': round(engagement_rate, 2)
                            })
            
            df = pd.DataFrame(engagement_data)
            
            if not df.empty:
                # Sort by engagement rate
                df = df.sort_values('engagement_rate', ascending=False).head(20)
                
                # Add performance labels
                def get_engagement_label(rate):
                    if rate > 10:
                        return 'Excellent'
                    elif rate > 5:
                        return 'Good'
                    elif rate > 2:
                        return 'Average'
                    else:
                        return 'Low'
                
                df['performance'] = df['engagement_rate'].apply(get_engagement_label)
            
            logger.info(f"Analyzed engagement for {len(df)} videos")
            return df
        
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return pd.DataFrame()
    
    # ==================== COMPETITOR ANALYSIS ====================
    
    def analyze_competitors(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Analyze tracked competitors
        
        Args:
            user_id: User identifier
        
        Returns:
            Competitor analysis
        """
        try:
            # Get competitors from database
            competitors = self.db.get_all_competitors(user_id)
            
            if not competitors:
                return {'message': 'No competitors tracked yet'}
            
            analysis = {
                'total_competitors': len(competitors),
                'analysis_date': datetime.now().isoformat(),
                'competitors': [],
                'summary': {}
            }
            
            # Analyze each competitor
            subscriber_counts = []
            avg_views_list = []
            
            for comp in competitors:
                subscriber_counts.append(comp.get('subscriber_count', 0))
                avg_views_list.append(comp.get('average_views', 0))
                
                # Calculate growth potential score
                growth_score = self._calculate_growth_score(comp)
                
                analysis['competitors'].append({
                    'channel_name': comp.get('channel_name', 'Unknown'),
                    'subscribers': comp.get('subscriber_count', 0),
                    'videos': comp.get('video_count', 0),
                    'average_views': comp.get('average_views', 0),
                    'growth_score': growth_score,
                    'last_checked': comp.get('last_checked', '')
                })
            
            # Calculate summary statistics
            if subscriber_counts:
                analysis['summary'] = {
                    'avg_subscribers': np.mean(subscriber_counts),
                    'median_subscribers': np.median(subscriber_counts),
                    'max_subscribers': np.max(subscriber_counts),
                    'min_subscribers': np.min(subscriber_counts),
                    'avg_views': np.mean(avg_views_list) if avg_views_list else 0
                }
            
            # Identify top competitors
            if analysis['competitors']:
                # Sort by growth score
                sorted_comps = sorted(
                    analysis['competitors'],
                    key=lambda x: x.get('growth_score', 0),
                    reverse=True
                )
                
                analysis['top_competitors'] = sorted_comps[:5]
                analysis['recommendations'] = self._generate_competitor_recommendations(
                    analysis['competitors']
                )
            
            logger.info(f"Analyzed {len(competitors)} competitors")
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing competitors: {e}")
            return {'error': str(e)}
    
    def _calculate_growth_score(self, competitor: Dict[str, Any]) -> float:
        """
        Calculate growth potential score for a competitor
        
        Args:
            competitor: Competitor data
        
        Returns:
            Growth score (0-100)
        """
        try:
            score = 0
            
            # Subscriber count factor (smaller channels have more growth potential)
            subs = competitor.get('subscriber_count', 0)
            if subs < 10000:
                score += 30
            elif subs < 100000:
                score += 20
            elif subs < 1000000:
                score += 10
            
            # Engagement factor (views per subscriber)
            avg_views = competitor.get('average_views', 0)
            if subs > 0:
                views_per_sub = avg_views / subs
                if views_per_sub > 1:
                    score += 40
                elif views_per_sub > 0.5:
                    score += 30
                elif views_per_sub > 0.1:
                    score += 20
                else:
                    score += 10
            
            # Content volume factor
            videos = competitor.get('video_count', 0)
            if videos > 100:
                score += 20
            elif videos > 50:
                score += 15
            elif videos > 10:
                score += 10
            
            # Recent activity factor (simplified)
            last_checked = competitor.get('last_checked', '')
            if last_checked:
                try:
                    last_date = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
                    days_since = (datetime.now(last_date.tzinfo) - last_date).days
                    
                    if days_since < 7:
                        score += 10
                    elif days_since < 30:
                        score += 5
                except:
                    score += 5
            
            return min(score, 100)
        
        except Exception:
            return 50
    
    def _generate_competitor_recommendations(self, competitors: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on competitor analysis
        
        Args:
            competitors: List of competitor data
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not competitors:
            return recommendations
        
        # Find top performer
        top_competitor = max(competitors, key=lambda x: x.get('average_views', 0))
        
        recommendations.append(
            f"Study {top_competitor.get('channel_name')} - they have the highest average views "
            f"({top_competitor.get('average_views', 0):,} views per video)"
        )
        
        # Find fast-growing channels
        small_growing = [
            c for c in competitors 
            if c.get('subscribers', 0) < 50000 and c.get('growth_score', 0) > 70
        ]
        
        if small_growing:
            recommendations.append(
                f"Monitor {len(small_growing)} small but fast-growing channels for emerging trends"
            )
        
        # Analyze subscriber distribution
        subs_counts = [c.get('subscribers', 0) for c in competitors]
        avg_subs = np.mean(subs_counts) if subs_counts else 0
        
        if avg_subs > 100000:
            recommendations.append(
                "You're tracking established channels. Consider adding some smaller, "
                "emerging channels to identify growth opportunities."
            )
        
        return recommendations[:3]  # Return top 3 recommendations
    
    # ==================== EXPORT FUNCTIONALITY ====================
    
    def get_export_data(self, user_id: str = "default") -> pd.DataFrame:
        """
        Prepare comprehensive data for export
        
        Args:
            user_id: User identifier
        
        Returns:
            DataFrame with all analytics data
        """
        try:
            export_data = []
            
            # Get trends data
            trends_df = self.get_trends(user_id, days=30)
            if not trends_df.empty:
                trends_df['data_type'] = 'trend'
                export_data.append(trends_df)
            
            # Get format analysis
            formats_df = self.get_top_formats(user_id)
            if not formats_df.empty:
                formats_df['data_type'] = 'format_analysis'
                export_data.append(formats_df)
            
            # Get engagement data
            engagement_df = self.get_video_engagement(user_id)
            if not engagement_df.empty:
                engagement_df['data_type'] = 'engagement_analysis'
                export_data.append(engagement_df)
            
            # Get overall metrics
            metrics = self.get_overall_metrics(user_id)
            if metrics and 'error' not in metrics:
                metrics_df = pd.DataFrame([metrics])
                metrics_df['data_type'] = 'overall_metrics'
                export_data.append(metrics_df)
            
            if export_data:
                # Combine all data
                combined_df = pd.concat(export_data, ignore_index=True)
                
                # Add export metadata
                combined_df['export_date'] = datetime.now().isoformat()
                combined_df['user_id'] = user_id
                
                logger.info(f"Prepared export data with {len(combined_df)} rows")
                return combined_df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error preparing export data: {e}")
            return pd.DataFrame()
    
    # ==================== INSIGHTS & RECOMMENDATIONS ====================
    
    def get_insights(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Generate actionable insights
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with insights
        """
        try:
            insights = {
                'date_generated': datetime.now().isoformat(),
                'user_id': user_id,
                'insights': [],
                'recommendations': [],
                'quick_wins': []
}
   # Get metrics
            metrics = self.get_overall_metrics(user_id)
            
            # Generate insights based on metrics
            avg_vph = metrics.get('average_vph', 0)
            success_rate = metrics.get('success_rate_percent', 0)
            engagement_rate = metrics.get('average_engagement_rate', 0)
            
            # VPH insights
            if avg_vph < 50:
                insights['insights'].append("Your average VPH is below 50, indicating low virality potential")
                insights['recommendations'].append("Focus on trending topics with higher search volume")
            elif avg_vph > 200:
                insights['insights'].append("Excellent VPH! You're finding highly viral content")
                insights['quick_wins'].append("Replicate the format of your highest VPH videos")
            
            # Success rate insights
            if success_rate < 10:
                insights['insights'].append(f"Only {success_rate:.1f}% of analyzed videos are successful (VPH > 100)")
                insights['recommendations'].append("Apply stricter filters to focus on proven performers")
            elif success_rate > 30:
                insights['insights'].append(f"Great success rate! {success_rate:.1f}% of analyzed videos are successful")
                insights['quick_wins'].append("Scale your research using your current filter settings")
            
            # Engagement insights
            if engagement_rate < 2:
                insights['insights'].append("Low engagement rates suggest content may not be resonating with audiences")
                insights['recommendations'].append("Look for content with higher like-to-view ratios")
            elif engagement_rate > 5:
                insights['insights'].append("High engagement rates indicate strong audience connection")
                insights['quick_wins'].append("Analyze comments on high-engagement videos for content ideas")
            
            # Format insights
            formats_df = self.get_top_formats(user_id)
            if not formats_df.empty:
                best_format = formats_df.loc[formats_df['average_vph'].idxmax()]
                insights['insights'].append(
                    f"{best_format['format']} videos have the highest average VPH ({best_format['average_vph']:.1f})"
                )
                insights['recommendations'].append(
                    f"Prioritize {best_format['format'].lower()} content in your strategy"
                )
            
            # Add generic insights if we don't have enough data
            if len(insights['insights']) < 3:
                insights['insights'].extend([
                    "Consider tracking competitors to benchmark performance",
                    "Regular analysis helps identify emerging trends early",
                    "Combine quantitative metrics with qualitative analysis"
                ])
            
            logger.info(f"Generated {len(insights['insights'])} insights")
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'error': str(e),
                'date_generated': datetime.now().isoformat()
            }
    
    def daily_insights(self, user_id: str = "default") -> List[str]:
        """
        Get daily insights (simple version)
        
        Args:
            user_id: User identifier
        
        Returns:
            List of insight messages
        """
        try:
            insights = []
            
            # Get today's date
            today = datetime.now().date()
            
            # Check for recent searches
            searches = self.db.get_search_history(limit=10)
            if searches:
                recent_search = searches[0]
                search_date = datetime.fromisoformat(
                    recent_search.get('created_at').replace('Z', '+00:00')
                ).date()
                
                if search_date == today:
                    insights.append(f"Today's research: '{recent_search.get('query')}'")
                else:
                    days_ago = (today - search_date).days
                    insights.append(f"Last research was {days_ago} days ago")
            else:
                insights.append("No research data yet. Start by searching for videos!")
            
            # Get competitor count
            competitors = self.db.get_all_competitors(user_id)
            if competitors:
                insights.append(f"Tracking {len(competitors)} competitors")
            
            # Get performance label
            metrics = self.get_overall_metrics(user_id)
            performance = metrics.get('performance_label', 'Unknown')
            insights.append(f"Current performance: {performance}")
            
            return insights[:5]  # Return top 5 insights
        
        except Exception as e:
            logger.error(f"Error getting daily insights: {e}")
            return ["Unable to generate insights at this time"]
```
