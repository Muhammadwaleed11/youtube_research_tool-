import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger("database")
logger.setLevel(logging.INFO)


class Database:
    """
    SQLite database wrapper for YouTube Research Tool
    """
    
    def __init__(self, db_path: str = "data/youtube_research.db"):
        """
        Initialize database connection and create tables if they don't exist
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()
        
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _create_tables(self):
        """
        Create all necessary tables if they don't exist
        """
        # Searches table - stores search history
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            params TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_count INTEGER DEFAULT 0
        )
        ''')
        
        # Results table - stores individual video results
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id INTEGER,
            video_id TEXT,
            title TEXT,
            channel_id TEXT,
            channel_title TEXT,
            channel_subscribers INTEGER,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            duration_seconds INTEGER,
            published_at TEXT,
            vph REAL,
            outlier_multiplier REAL,
            url TEXT,
            thumbnail_url TEXT,
            FOREIGN KEY (search_id) REFERENCES searches (id) ON DELETE CASCADE
        )
        ''')
        
        # Competitors table - stores tracked competitors
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS competitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            channel_id TEXT UNIQUE,
            channel_name TEXT,
            channel_url TEXT,
            description TEXT,
            subscriber_count INTEGER,
            video_count INTEGER,
            view_count INTEGER,
            average_views REAL,
            country TEXT,
            thumbnail_url TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tracking_active BOOLEAN DEFAULT 1
        )
        ''')
        
        # Competitor updates table - stores historical data
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS competitor_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competitor_id INTEGER,
            check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            subscriber_count INTEGER,
            video_count INTEGER,
            view_count INTEGER,
            average_views REAL,
            new_videos INTEGER DEFAULT 0,
            viral_videos TEXT,
            FOREIGN KEY (competitor_id) REFERENCES competitors (id) ON DELETE CASCADE
        )
        ''')
        
        # Alerts table - stores notifications/alerts
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            competitor_id INTEGER,
            alert_type TEXT,
            alert_message TEXT,
            alert_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read BOOLEAN DEFAULT 0,
            FOREIGN KEY (competitor_id) REFERENCES competitors (id) ON DELETE CASCADE
        )
        ''')
        
        # Analytics table - stores performance metrics
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            metric_date DATE,
            metric_type TEXT,
            metric_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    # ==================== SEARCHES ====================
    
    def save_search(self, query: str, params: Dict[str, Any]) -> int:
        """
        Save a search to the database
        
        Args:
            query: Search query string
            params: Search parameters as dictionary
        
        Returns:
            Search ID
        """
        try:
            params_json = json.dumps(params)
            
            self.cursor.execute('''
            INSERT INTO searches (query, params, created_at)
            VALUES (?, ?, datetime('now'))
            ''', (query, params_json))
            
            search_id = self.cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Search saved: ID={search_id}, Query='{query}'")
            return search_id
        
        except Exception as e:
            logger.error(f"Failed to save search: {e}")
            raise
    
    def save_results(self, search_id: int, results_df: pd.DataFrame):
        """
        Save search results to database
        
        Args:
            search_id: Search ID from save_search
            results_df: DataFrame containing results
        """
        try:
            # Update search result count
            self.cursor.execute('''
            UPDATE searches 
            SET result_count = ? 
            WHERE id = ?
            ''', (len(results_df), search_id))
            
            # Insert each result
            for _, row in results_df.iterrows():
                self.cursor.execute('''
                INSERT INTO results (
                    search_id, video_id, title, channel_id, channel_title,
                    channel_subscribers, views, likes, comments, duration_seconds,
                    published_at, vph, outlier_multiplier, url, thumbnail_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    search_id,
                    row.get('video_id'),
                    row.get('title', '')[:500],  # Limit title length
                    row.get('channel_id'),
                    row.get('channel_title', '')[:200],
                    row.get('channel_subscribers'),
                    row.get('views', 0),
                    row.get('likes', 0),
                    row.get('comments', 0),
                    row.get('duration_seconds', 0),
                    row.get('published_at'),
                    row.get('VPH') or row.get('vph'),
                    row.get('OutlierMultiplier') or row.get('outlier_multiplier', 1.0),
                    row.get('url'),
                    row.get('thumbnail_url')
                ))
            
            self.conn.commit()
            logger.info(f"Saved {len(results_df)} results for search {search_id}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent search history
        
        Args:
            limit: Maximum number of searches to return
        
        Returns:
            List of search dictionaries
        """
        try:
            self.cursor.execute('''
            SELECT id, query, params, created_at, result_count
            FROM searches
            ORDER BY created_at DESC
            LIMIT ?
            ''', (limit,))
            
            searches = []
            for row in self.cursor.fetchall():
                searches.append({
                    'id': row[0],
                    'query': row[1],
                    'params': json.loads(row[2]) if row[2] else {},
                    'created_at': row[3],
                    'result_count': row[4]
                })
            return searches
        
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []
    
    def get_results_for_search(self, search_id: int) -> Optional[pd.DataFrame]:
        """
        Get results for a specific search
        
        Args:
            search_id: Search ID
        
        Returns:
            DataFrame of results or None if not found
        """
        try:
            self.cursor.execute('''
            SELECT 
                video_id, title, channel_id, channel_title,
                channel_subscribers, views, likes, comments,
                duration_seconds, published_at, vph, outlier_multiplier, url
            FROM results
            WHERE search_id = ?
            ORDER BY vph DESC
            ''', (search_id,))
            
            columns = [
                'video_id', 'title', 'channel_id', 'channel_title',
                'channel_subscribers', 'views', 'likes', 'comments',
                'duration_seconds', 'published_at', 'vph', 'outlier_multiplier', 'url'
            ]
            
            rows = self.cursor.fetchall()
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=columns)
            return df
        
        except Exception as e:
            logger.error(f"Failed to get results for search {search_id}: {e}")
            return None
    
    # ==================== COMPETITORS ====================
    
    def add_competitor(self, **kwargs) -> bool:
        """
        Add a new competitor to track
        
        Args:
            **kwargs: Competitor details
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if competitor already exists
            channel_id = kwargs.get('channel_id')
            if channel_id:
                self.cursor.execute(
                    'SELECT id FROM competitors WHERE channel_id = ?',
                    (channel_id,)
                )
                if self.cursor.fetchone():
                    logger.warning(f"Competitor already exists: {channel_id}")
                    return False
            
            # Insert new competitor
            columns = []
            values = []
            placeholders = []
            
            for key, value in kwargs.items():
                columns.append(key)
                values.append(value)
                placeholders.append('?')
            
            sql = f'''
            INSERT INTO competitors ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            '''
            
            self.cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Competitor added: {kwargs.get('channel_name', 'Unknown')}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add competitor: {e}")
            return False
    
    def update_competitor(self, channel_id: str, **kwargs) -> bool:
        """
        Update competitor information
        
        Args:
            channel_id: YouTube channel ID
            **kwargs: Fields to update
        
        Returns:
            True if successful
        """
        try:
            # Build SET clause
            set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
            set_clause += ", last_checked = datetime('now')"
            
            values = list(kwargs.values())
            values.append(channel_id)
            
            sql = f'''
            UPDATE competitors 
            SET {set_clause}
            WHERE channel_id = ?
            '''
            
            self.cursor.execute(sql, values)
            self.conn.commit()
            
            # Save historical update
            competitor = self.get_competitor_by_channel_id(channel_id)
            if competitor:
                self.save_competitor_update(competitor['id'], kwargs)
            
            logger.info(f"Competitor updated: {channel_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update competitor {channel_id}: {e}")
            return False
    
    def save_competitor_update(self, competitor_id: int, data: Dict[str, Any]):
        """
        Save historical update for competitor
        
        Args:
            competitor_id: Competitor ID
            data: Update data
        """
        try:
            viral_videos = json.dumps(data.get('viral_videos', [])) if data.get('viral_videos') else None
            
            self.cursor.execute('''
            INSERT INTO competitor_updates 
            (competitor_id, subscriber_count, video_count, view_count, 
             average_views, new_videos, viral_videos)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                competitor_id,
                data.get('subscriber_count'),
                data.get('video_count'),
                data.get('view_count'),
                data.get('average_views'),
                data.get('new_videos', 0),
                viral_videos
            ))
            
            self.conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to save competitor update: {e}")
    
    def get_all_competitors(self, user_id: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all tracked competitors
        
        Args:
            user_id: User ID (default: 'default')
            limit: Maximum number to return
        
        Returns:
            List of competitor dictionaries
        """
        try:
            self.cursor.execute('''
            SELECT 
                id, channel_id, channel_name, channel_url,
                subscriber_count, video_count, view_count, average_views,
                added_date, last_checked, tracking_active
            FROM competitors
            WHERE user_id = ? AND tracking_active = 1
            ORDER BY added_date DESC
            LIMIT ?
            ''', (user_id, limit))
            
            competitors = []
            for row in self.cursor.fetchall():
                competitors.append(dict(row))
            
            return competitors
        
        except Exception as e:
            logger.error(f"Failed to get competitors: {e}")
            return []
    
    def get_competitor_by_channel_id(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get competitor by channel ID
        
        Args:
            channel_id: YouTube channel ID
        
        Returns:
            Competitor dictionary or None
        """
        try:
            self.cursor.execute('''
            SELECT * FROM competitors WHERE channel_id = ?
            ''', (channel_id,))
            
            row = self.cursor.fetchone()
            return dict(row) if row else None
        
        except Exception as e:
            logger.error(f"Failed to get competitor {channel_id}: {e}")
            return None
    
    def get_competitor_updates(self, competitor_id: int, days: int = 30) -> pd.DataFrame:
        """
        Get historical updates for a competitor
        
        Args:
            competitor_id: Competitor ID
            days: Number of days to look back
        
        Returns:
            DataFrame of updates
        """
        try:
            self.cursor.execute('''
            SELECT 
                check_date, subscriber_count, video_count, view_count,
                average_views, new_videos
            FROM competitor_updates
            WHERE competitor_id = ? 
            AND check_date >= datetime('now', ?)
            ORDER BY check_date ASC
            ''', (competitor_id, f'-{days} days'))
            
            columns = ['check_date', 'subscriber_count', 'video_count', 
                      'view_count', 'average_views', 'new_videos']
            
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=columns)
            return df
        
        except Exception as e:
            logger.error(f"Failed to get competitor updates: {e}")
            return pd.DataFrame()
    
    # ==================== ALERTS ====================
    
    def add_alert(self, **kwargs) -> int:
        """
        Add a new alert
        
        Args:
            **kwargs: Alert details
        
        Returns:
            Alert ID
        """
        try:
            columns = []
            values = []
            placeholders = []
            
            for key, value in kwargs.items():
                columns.append(key)
                values.append(value)
                placeholders.append('?')
            
            sql = f'''
            INSERT INTO alerts ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            '''
          self.cursor.execute(sql, values)
            alert_id = self.cursor.lastrowid
            self.conn.commit()
            
            logger.info(f"Alert added: ID={alert_id}")
            return alert_id
        
        except Exception as e:
            logger.error(f"Failed to add alert: {e}")
            return -1
    
    def get_recent_alerts(self, user_id: str = "default", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            user_id: User ID
            limit: Maximum alerts to return
        
        Returns:
            List of alert dictionaries
        """
        try:
            self.cursor.execute('''
            SELECT 
                a.id, a.alert_type, a.alert_message, a.created_at,
                a.is_read, c.channel_name
            FROM alerts a
            LEFT JOIN competitors c ON a.competitor_id = c.id
            WHERE a.user_id = ?
            ORDER BY a.created_at DESC
            LIMIT ?
            ''', (user_id, limit))
            
            alerts = []
            for row in self.cursor.fetchall():
                alerts.append(dict(row))
            
            return alerts
        
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def mark_alert_read(self, alert_id: int) -> bool:
        """
        Mark an alert as read
        
        Args:
            alert_id: Alert ID
        
        Returns:
            True if successful
        """
        try:
            self.cursor.execute('''
            UPDATE alerts SET is_read = 1 WHERE id = ?
            ''', (alert_id,))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            logger.error(f"Failed to mark alert {alert_id} as read: {e}")
            return False
    
    # ==================== ANALYTICS ====================
    
    def save_analytics(self, user_id: str, metric_type: str, metric_value: float):
        """
        Save analytics metric
        
        Args:
            user_id: User ID
            metric_type: Type of metric
            metric_value: Metric value
        """
        try:
            self.cursor.execute('''
            INSERT INTO analytics (user_id, metric_date, metric_type, metric_value)
            VALUES (?, date('now'), ?, ?)
            ''', (user_id, metric_type, metric_value))
            
            self.conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to save analytics: {e}")
    
    def get_analytics(self, user_id: str, metric_type: str, days: int = 30) -> pd.DataFrame:
        """
        Get analytics data
        
        Args:
            user_id: User ID
            metric_type: Type of metric
            days: Number of days to look back
        
        Returns:
            DataFrame of analytics
        """
        try:
            self.cursor.execute('''
            SELECT 
                metric_date, metric_value, created_at
            FROM analytics
            WHERE user_id = ? 
            AND metric_type = ?
            AND metric_date >= date('now', ?)
            ORDER BY metric_date ASC
            ''', (user_id, metric_type, f'-{days} days'))
            
            columns = ['metric_date', 'metric_value', 'created_at']
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=columns)
            return df
        
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return pd.DataFrame()
    
    # ==================== UTILITY ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary of statistics
        """
        try:
            stats = {}
            
            # Count searches
            self.cursor.execute('SELECT COUNT(*) FROM searches')
            stats['total_searches'] = self.cursor.fetchone()[0]
            
            # Count results
            self.cursor.execute('SELECT COUNT(*) FROM results')
            stats['total_results'] = self.cursor.fetchone()[0]
            
            # Count competitors
            self.cursor.execute('SELECT COUNT(*) FROM competitors')
            stats['total_competitors'] = self.cursor.fetchone()[0]
            
            # Count alerts
            self.cursor.execute('SELECT COUNT(*) FROM alerts WHERE is_read = 0')
            stats['unread_alerts'] = self.cursor.fetchone()[0]
            
            # Recent activity
            self.cursor.execute('''
            SELECT COUNT(*) FROM searches 
            WHERE created_at >= datetime('now', '-1 day')
            ''')
            stats['searches_today'] = self.cursor.fetchone()[0]
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# For backward compatibility
def get_db_stats(db_path: str = "data/youtube_research.db") -> Dict[str, Any]:
    """Quick function to get database statistics"""
    try:
        db = Database(db_path)
        return db.get_stats()
    except Exception as e:
        logger.error(f"Failed to get DB stats: {e}")
        return {}
```
