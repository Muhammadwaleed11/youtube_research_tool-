import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import modules
try:
    from modules.youtube_api import YouTubeAPI, YoutubeAPIError
    from modules.database import Database
    from modules.filters import apply_filters_to_df
    from modules.utils import calculate_vph, format_number, parse_duration
    from modules.competitor_tracker import CompetitorTracker
    from modules.analytics import AnalyticsService
except ImportError as e:
    st.error(f"Module import error: {e}. Make sure modules folder exists.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YouTube Research Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'services' not in st.session_state:
    st.session_state.services = {}

# Sidebar navigation
st.sidebar.title("🎯 YouTube Research Pro")

# API Key input
api_key = st.sidebar.text_input(
    "YouTube API Key",
    value=st.session_state.api_key,
    type="password",
    help="Enter your YouTube Data API v3 key"
)
if api_key != st.session_state.api_key:
    st.session_state.api_key = api_key
    st.session_state.services = {}  # Reset services

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home Dashboard", "🔍 Research Tool", "🎯 Competitor Tracking", "📊 Analytics"],
    index=["🏠 Home Dashboard", "🔍 Research Tool", "🎯 Competitor Tracking", "📊 Analytics"].index(
        st.session_state.get('current_page', '🏠 Home Dashboard')
    )
)

# Update session state
st.session_state.current_page = page

# Initialize services
def init_services():
    """Initialize all services (API, DB, Tracker, Analytics)"""
    if not st.session_state.api_key:
        return None
    
    if 'services' in st.session_state and st.session_state.services:
        return st.session_state.services
    
    try:
        # Initialize YouTube API
        youtube = YouTubeAPI(api_key=st.session_state.api_key)
        
        # Initialize Database
        db = Database("data/youtube_research.db")
        
        # Initialize Competitor Tracker
        tracker = CompetitorTracker(db, youtube)
        
        # Initialize Analytics
        analytics = AnalyticsService(db)
        
        services = {
            'youtube': youtube,
            'db': db,
            'tracker': tracker,
            'analytics': analytics
        }
        
        st.session_state.services = services
        return services
    except Exception as e:
        st.sidebar.error(f"Failed to initialize services: {e}")
        return None

# Initialize services
services = init_services()

# Display current page
st.session_state.page = page

# Home Page
if page == "🏠 Home Dashboard":
    st.title("🏠 Home Dashboard")
    st.markdown("Welcome to YouTube Research Pro - Your Complete Research Companion")
    
    if services:
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Searches", "142", "+12 today")
        with col2:
            st.metric("Tracked Competitors", "8", "+2 this week")
        with col3:
            st.metric("Avg VPH", "245", "+15%")
        with col4:
            st.metric("API Quota", "3,450/10,000", "34.5%")
        
        st.markdown("---")
        
        # Recent Activity
        st.subheader("📊 Recent Activity")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recent Searches**")
            try:
                searches = services['db'].get_search_history(limit=5)
                for search in searches:
                    st.write(f"• {search['query']} ({search['created_at'][:10]})")
            except:
                st.info("No recent searches")
        
        with col2:
            st.markdown("**Tracked Competitors**")
            try:
                competitors = services['tracker'].get_all_competitors(limit=5)
                for comp in competitors:
                    st.write(f"• {comp['channel_name']} ({comp['subscriber_count']:,} subs)")
            except:
                st.info("No competitors tracked")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Quick Search"):
                st.session_state.current_page = "🔍 Research Tool"
                st.experimental_rerun()
        
        with col2:
            if st.button("🎯 Add Competitor"):
                st.session_state.current_page = "🎯 Competitor Tracking"
                st.experimental_rerun()
        
        with col3:
            if st.button("📊 View Analytics"):
                st.session_state.current_page = "📊 Analytics"
                st.experimental_rerun()
    
    else:
        st.warning("Please enter your YouTube API key in the sidebar to get started.")

# Research Tool Page
elif page == "🔍 Research Tool":
    st.title("🔍 Research Tool")
    
    if not services:
        st.warning("Please enter YouTube API key in sidebar")
        st.stop()
    
    # Search Form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search keywords", value=st.session_state.get('last_search', ''))
        with col2:
            max_results = st.number_input("Max results", min_value=10, max_value=100, value=50)
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_views = st.number_input("Minimum views", value=1000)
            max_views = st.number_input("Maximum views", value=0)
        
        with col2:
            min_subs = st.number_input("Minimum subscribers", value=0)
            max_subs = st.number_input("Maximum subscribers", value=0)
        
        with col3:
            duration = st.selectbox("Video duration", ["Any", "Short (<4m)", "Medium (4-20m)", "Long (>20m)"])
            days_back = st.selectbox("Published in last", ["Any", "1 day", "7 days", "30 days", "90 days"])
        
        submitted = st.form_submit_button("Search")
    
    if submitted and query:
        st.session_state.last_search = query
        
        with st.spinner("Searching YouTube..."):
            try:
                # Calculate published after
                published_after = None
                if days_back != "Any":
                    days_map = {"1 day": 1, "7 days": 7, "30 days": 30, "90 days": 90}
                    days = days_map.get(days_back, 0)
                    if days:
                        published_after = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
                
                # Search videos
                results = services['youtube'].search_videos(
                    query=query,
                    max_results=max_results,
                    published_after=published_after,
                    duration={"Short (<4m)": "short", "Medium (4-20m)": "medium", "Long (>20m)": "long"}.get(duration)
                )
                
                if results:
                    df = pd.DataFrame(results)
                    
                    # Calculate VPH
                    df['VPH'] = df.apply(
                        lambda row: calculate_vph(row['views'], row['published_at']),
                        axis=1
                    )
                    
                    # Apply filters
                    df = apply_filters_to_df(
                        df,
                        min_views=min_views if min_views > 0 else None,
                        max_views=max_views if max_views > 0 else None,
                        min_subs=min_subs if min_subs > 0 else None,
                        max_subs=max_subs if max_subs > 0 else None,
                        duration_choice=duration
                    )
                    
                    # Display results
                    st.subheader(f"Results: {len(df)} videos")
                    
                    # Data table
                    display_cols = ['title', 'channel_title', 'views', 'VPH', 'published_at', 'url']
                    display_df = df[display_cols].copy()
                    display_df['views'] = display_df['views'].apply(format_number)
                    display_df['VPH'] = display_df['VPH'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"youtube_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
                    # Save to DB
                    try:
                        search_id = services['db'].save_search(query=query, params={
                            'max_results': max_results,
                            'filters': {
                                'min_views': min_views,
                                'max_views': max_views,
                                'min_subs': min_subs,
                                'max_subs': max_subs,
                                'duration': duration,
                                'days_back': days_back
                            }
                        })
                        services['db'].save_results(search_id, df)
                        st.success("Search saved to database")
                    except Exception as e:
                        st.warning(f"Could not save to DB: {e}")
                
                else:
                    st.info("No results found")
                    
            except Exception as e:
                st.error(f"Search error: {e}")

# Competitor Tracking Page
elif page == "🎯 Competitor Tracking":
    st.title("🎯 Competitor Tracking")
    
    if not services:
        st.warning("Please enter YouTube API key in sidebar")
        st.stop()
    
    tracker = services['tracker']
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Add Competitor", "Tracked Competitors", "Alerts"])
    
    with tab1:
        st.subheader("Add New Competitor")
        
        channel_url = st.text_input("Enter YouTube Channel URL")
        if st.button("Add Competitor") and channel_url:
            with st.spinner("Adding competitor..."):
                try:
                    success = tracker.add_competitor(channel_url, user_id="default")
                    if success:
                        st.success("Competitor added successfully!")
                    else:
                        st.error("Failed to add competitor")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Tracked Competitors")
        
        try:
            competitors = tracker.get_all_competitors()
            if competitors:
                comp_df = pd.DataFrame(competitors)
                st.dataframe(comp_df, use_container_width=True)
                
                # Select competitor for details
                selected = st.selectbox("Select competitor for details", comp_df['channel_name'].tolist())
                
                if selected:
                    competitor = comp_df[comp_df['channel_name'] == selected].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Subscribers", format_number(competitor['subscriber_count']))
                    with col2:
                        st.metric("Videos", competitor['video_count'])
                    with col3:
                        st.metric("Avg Views", format_number(competitor['average_views']))
                    with col4:
                        st.metric("Last Checked", competitor['last_checked'][:10])
                    
                    # Check for updates
                    if st.button("Check for Updates"):
                        with st.spinner("Checking updates..."):
                            updates = tracker.get_competitor_updates(competitor['channel_id'])
                            if updates:
                                st.info(f"Found {updates.get('new_videos', 0)} new videos")
                                if updates.get('viral_videos'):
                                    st.warning(f"Viral video detected: {updates['viral_videos'][0]['title']}")
            else:
                st.info("No competitors tracked yet")
        except Exception as e:
            st.error(f"Error loading competitors: {e}")
    
    with tab3:
        st.subheader("Alerts & Notifications")
        
        if st.button("Check for Alerts"):
            with st.spinner("Checking alerts..."):
                try:
                    tracker.send_alerts(user_id="default")
                    alerts = services['db'].get_recent_alerts(limit=10)
                    
                    if alerts:
                        for alert in alerts:
                            st.warning(f"🔔 {alert['alert_message']} ({alert['created_at'][:10]})")
                    else:
                        st.info("No new alerts")
                except Exception as e:
                    st.error(f"Error checking alerts: {e}")

# Analytics Page
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")
    
    if not services:
        st.warning("Please enter YouTube API key in sidebar")
        st.stop()
    
    analytics = services['analytics']
    
    # Overall metrics
    st.subheader("Overall Metrics")
    metrics = analytics.get_overall_metrics(user_id="default")
    st.dataframe(metrics, use_container_width=True)
    
    # Trends
    st.subheader("Trends")
    trends = analytics.get_trends(user_id="default")
    
    if not trends.empty:
        import plotly.express as px
        
        fig1 = px.line(trends, x='date', y='views', title='Views Over Time')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.line(trends, x='date', y='engagement_rate', title='Engagement Rate Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top formats
    st.subheader("Top Performing Formats")
    formats = analytics.get_top_formats(user_id="default")
    st.dataframe(formats, use_container_width=True)
    
    # Export
    st.subheader("Export Data")
    if st.button("Export All Analytics"):
        export_df = analytics.get_export_data(user_id="default")
        csv = export_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("YouTube Research Pro v1.0")
if st.sidebar.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
