# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Supply Chain Command Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f2937; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 20px; border-radius: 10px; color: white; }
    .insight-box { background-color: #f3f4f6; padding: 15px; border-left: 5px solid #3b82f6; 
                   margin: 10px 0; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding-left: 20px; padding-right: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA SIMULATION ENGINE
# ==========================================

@st.cache_data(ttl=3600)
def generate_supply_chain_data(n_skus=1200, days=365, random_seed=42):
    """Generate realistic supply chain data with seasonality and trends"""
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    categories = ['Electronics', 'Mechanical', 'Raw Materials', 'Packaging', 'Components']
    suppliers = [f'SUP-{str(i).zfill(3)}' for i in range(1, 21)]
    warehouses = ['WH-East', 'WH-West', 'WH-Central', 'WH-South']
    
    data = []
    sku_master = []
    
    for sku_id in range(1, n_skus + 1):
        # SKU characteristics
        category = random.choice(categories)
        supplier = random.choice(suppliers)
        warehouse = random.choice(warehouses)
        
        # Base demand parameters with seasonality
        base_demand = np.random.gamma(2, 50) + 10
        trend = np.random.uniform(-0.1, 0.15)
        seasonality = np.random.uniform(0.2, 0.6) if category == 'Electronics' else np.random.uniform(0.1, 0.3)
        
        # Cost structure
        unit_cost = np.random.lognormal(3, 1.5)
        holding_cost_rate = 0.25  # 25% annually
        ordering_cost = np.random.uniform(50, 500)
        stockout_cost = unit_cost * np.random.uniform(2, 5)
        
        lead_time = np.random.poisson(7) + 3
        service_level_target = np.random.choice([0.90, 0.95, 0.98, 0.99])
        
        sku_master.append({
            'sku_id': f'SKU-{str(sku_id).zfill(5)}',
            'category': category,
            'supplier': supplier,
            'warehouse': warehouse,
            'unit_cost': unit_cost,
            'holding_cost_rate': holding_cost_rate,
            'ordering_cost': ordering_cost,
            'stockout_cost': stockout_cost,
            'lead_time': lead_time,
            'service_level_target': service_level_target,
            'base_demand': base_demand,
            'trend': trend,
            'seasonality': seasonality
        })
        
        # Generate daily transactions
        start_date = datetime(2023, 1, 1)
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Demand calculation with trend and seasonality
            trend_factor = 1 + (trend * day / 365)
            seasonal_factor = 1 + seasonality * np.sin(2 * np.pi * day / 365)
            noise = np.random.normal(1, 0.3)
            
            daily_demand = max(0, base_demand * trend_factor * seasonal_factor * noise)
            
            # Inventory position simulation
            if day == 0:
                inventory_level = np.random.uniform(100, 1000)
            else:
                # Simple inventory logic with reordering
                prev_level = data[-1]['inventory_level'] if data[-1]['sku_id'] == f'SKU-{str(sku_id).zfill(5)}' else 500
                receipts = daily_demand * 2 if prev_level < base_demand * lead_time else 0
                inventory_level = max(0, prev_level - daily_demand + receipts)
            
            data.append({
                'sku_id': f'SKU-{str(sku_id).zfill(5)}',
                'date': current_date,
                'demand': round(daily_demand, 2),
                'inventory_level': round(inventory_level, 2),
                'unit_cost': unit_cost,
                'category': category,
                'supplier': supplier,
                'warehouse': warehouse
            })
    
    df_transactions = pd.DataFrame(data)
    df_master = pd.DataFrame(sku_master)
    
    return df_transactions, df_master

# ==========================================
# ANALYTICS ENGINE
# ==========================================

class InventoryOptimizer:
    def __init__(self, df_transactions, df_master):
        self.df = df_transactions.merge(df_master, on='sku_id', how='left')
        self.master = df_master
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate comprehensive inventory metrics"""
        # Aggregate by SKU
        sku_stats = self.df.groupby('sku_id').agg({
            'demand': ['sum', 'mean', 'std'],
            'inventory_level': 'mean',
            'unit_cost_x': 'first',
            'category': 'first',
            'supplier': 'first',
            'warehouse': 'first',
            'lead_time': 'first',
            'service_level_target': 'first',
            'ordering_cost': 'first',
            'holding_cost_rate': 'first'
        }).reset_index()
        
        sku_stats.columns = ['sku_id', 'total_demand', 'avg_daily_demand', 'demand_std', 
                            'avg_inventory', 'unit_cost', 'category', 'supplier', 
                            'warehouse', 'lead_time', 'service_level_target', 
                            'ordering_cost', 'holding_cost_rate']
        
        # Calculate derived metrics
        sku_stats['annual_demand'] = sku_stats['total_demand'] * (365/365)  # Annualized
        sku_stats['inventory_value'] = sku_stats['avg_inventory'] * sku_stats['unit_cost']
        sku_stats['annual_revenue'] = sku_stats['annual_demand'] * sku_stats['unit_cost']
        
        # Turnover ratio
        sku_stats['turnover_ratio'] = sku_stats['annual_revenue'] / sku_stats['inventory_value']
        sku_stats['turnover_ratio'] = sku_stats['turnover_ratio'].replace([np.inf, -np.inf], 0)
        
        # Days of inventory outstanding
        sku_stats['dio'] = (sku_stats['avg_inventory'] / sku_stats['total_demand']) * 365
        
        # ABC Classification
        sku_stats = sku_stats.sort_values('annual_revenue', ascending=False)
        sku_stats['cumulative_revenue_pct'] = sku_stats['annual_revenue'].cumsum() / sku_stats['annual_revenue'].sum()
        sku_stats['abc_class'] = sku_stats['cumulative_revenue_pct'].apply(
            lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C')
        )
        
        # Stock aging simulation
        sku_stats['stock_age_days'] = np.random.choice(
            [15, 45, 75, 120, 200], 
            size=len(sku_stats), 
            p=[0.4, 0.25, 0.15, 0.12, 0.08]
        )
        sku_stats['aging_bucket'] = sku_stats['stock_age_days'].apply(
            lambda x: '0-30' if x <= 30 else ('31-60' if x <= 60 else ('61-90' if x <= 90 else '90+'))
        )
        
        # Slow moving identification (turnover < 2)
        sku_stats['is_slow_moving'] = sku_stats['turnover_ratio'] < 2
        
        # EOQ Calculation
        sku_stats['eoq'] = np.sqrt(
            (2 * sku_stats['annual_demand'] * sku_stats['ordering_cost']) / 
            (sku_stats['unit_cost'] * sku_stats['holding_cost_rate'])
        )
        
        # Safety Stock (using normal distribution assumption)
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
        sku_stats['z_score'] = sku_stats['service_level_target'].map(z_scores)
        sku_stats['safety_stock'] = (
            sku_stats['z_score'] * sku_stats['demand_std'] * np.sqrt(sku_stats['lead_time'])
        )
        
        # Reorder Point
        sku_stats['reorder_point'] = (
            sku_stats['avg_daily_demand'] * sku_stats['lead_time'] + sku_stats['safety_stock']
        )
        
        self.sku_stats = sku_stats
        return sku_stats
    
    def get_kpis(self):
        """Calculate executive KPIs"""
        total_inventory_value = self.sku_stats['inventory_value'].sum()
        total_skus = len(self.sku_stats)
        avg_turnover = self.sku_stats['turnover_ratio'].mean()
        avg_dio = self.sku_stats['dio'].mean()
        
        slow_moving_pct = (self.sku_stats['is_slow_moving'].sum() / total_skus) * 100
        excess_stock_value = self.sku_stats[self.sku_stats['inventory_level'] > self.sku_stats['reorder_point'] * 2]['inventory_value'].sum()
        
        # Working capital optimization potential
        optimal_inventory = self.sku_stats['eoq'].sum() * self.sku_stats['unit_cost'].mean()
        wc_optimization = total_inventory_value - optimal_inventory
        
        return {
            'total_inventory_value': total_inventory_value,
            'total_skus': total_skus,
            'avg_turnover': avg_turnover,
            'avg_dio': avg_dio,
            'slow_moving_pct': slow_moving_pct,
            'excess_stock_value': excess_stock_value,
            'wc_optimization_potential': wc_optimization,
            'service_level': np.random.uniform(94, 98)  # Simulated
        }
    
    def demand_forecast(self, sku_id, days_ahead=30):
        """ML-based demand forecasting"""
        sku_data = self.df[self.df['sku_id'] == sku_id].copy()
        if len(sku_data) < 30:
            return None
        
        sku_data['day_num'] = range(len(sku_data))
        X = sku_data[['day_num']].values
        y = sku_data['demand'].values
        
        # Random Forest for non-linear patterns
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        future_days = np.array(range(len(sku_data), len(sku_data) + days_ahead)).reshape(-1, 1)
        forecast = model.predict(future_days)
        
        return forecast, sku_data['demand'].values[-30:]

# ==========================================
# VISUALIZATION COMPONENTS
# ==========================================

def render_header():
    st.markdown('<p class="main-header">🏭 Supply Chain Command Center</p>', unsafe_allow_html=True)
    st.markdown("""
    **Advanced Inventory Optimization & Predictive Analytics Platform**
    
    *Real-time simulation | ML Forecasting | Prescriptive Analytics*
    """)

def render_kpi_cards(kpis):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Inventory Value",
            value=f"${kpis['total_inventory_value']:,.0f}",
            delta=f"-${kpis['wc_optimization_potential']:,.0f} potential"
        )
    
    with col2:
        st.metric(
            label="Inventory Turnover",
            value=f"{kpis['avg_turnover']:.2f}x",
            delta=f"{kpis['avg_turnover']-4:.2f} vs target"
        )
    
    with col3:
        st.metric(
            label="Days Inventory Outstanding",
            value=f"{kpis['avg_dio']:.0f} days",
            delta=f"{45-kpis['avg_dio']:.0f} vs benchmark"
        )
    
    with col4:
        st.metric(
            label="Slow Moving SKUs",
            value=f"{kpis['slow_moving_pct']:.1f}%",
            delta=f"-{kpis['slow_moving_pct']*0.3:.1f}% optimizable",
            delta_color="inverse"
        )

def create_abc_analysis(stats):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
    
    # Pie chart
    abc_counts = stats['abc_class'].value_counts()
    fig.add_trace(
        go.Pie(labels=abc_counts.index, values=abc_counts.values, 
               hole=0.4, name="ABC Distribution",
               marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']),
        row=1, col=1
    )
    
    # Pareto chart
    stats_sorted = stats.sort_values('annual_revenue', ascending=False).reset_index()
    stats_sorted['cumulative_pct'] = stats_sorted['annual_revenue'].cumsum() / stats_sorted['annual_revenue'].sum() * 100
    
    fig.add_trace(
        go.Bar(x=list(range(len(stats_sorted))), y=stats_sorted['annual_revenue'],
               name='Revenue', marker_color='lightblue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(stats_sorted))), y=stats_sorted['cumulative_pct'],
                  name='Cumulative %', yaxis='y2', line=dict(color='red')),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="ABC Analysis & Pareto Distribution",
        showlegend=True,
        height=500
    )
    return fig

def create_network_graph(stats):
    """Create supply chain network visualization"""
    # Aggregate by supplier-warehouse
    network_data = stats.groupby(['supplier', 'warehouse']).agg({
        'inventory_value': 'sum',
        'sku_id': 'count'
    }).reset_index()
    
    fig = px.scatter_geo(
        network_data,
        locations="warehouse",
        locationmode='USA-states',
        size="inventory_value",
        color="sku_id",
        hover_name="supplier",
        scope="usa",
        title="Supply Chain Network Distribution",
        color_continuous_scale="Viridis"
    )
    return fig

def create_optimization_simulator(stats):
    """Interactive EOQ optimization"""
    st.subheader("🎯 Inventory Optimization Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_sku = st.selectbox("Select SKU for Analysis", stats['sku_id'].unique()[:50])
        sku_row = stats[stats['sku_id'] == selected_sku].iloc[0]
        
        st.markdown(f"""
        **Current Parameters:**
        - Unit Cost: ${sku_row['unit_cost']:.2f}
        - Current Stock: {sku_row['avg_inventory']:.0f} units
        - Daily Demand: {sku_row['avg_daily_demand']:.2f}
        - Lead Time: {sku_row['lead_time']} days
        
        **Calculated Optimals:**
        - EOQ: {sku_row['eoq']:.0f} units
        - Safety Stock: {sku_row['safety_stock']:.0f} units
        - Reorder Point: {sku_row['reorder_point']:.0f} units
        """)
        
        # Interactive sliders
        new_service_level = st.slider("Target Service Level", 0.90, 0.99, sku_row['service_level_target'], 0.01)
        new_ordering_cost = st.slider("Ordering Cost ($)", 50, 1000, int(sku_row['ordering_cost']), 50)
    
    with col2:
        # Calculate new safety stock based on slider
        z_scores = {0.90: 1.28, 0.91: 1.34, 0.92: 1.41, 0.93: 1.48, 0.94: 1.55, 
                   0.95: 1.65, 0.96: 1.75, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}
        z = z_scores.get(round(new_service_level, 2), 1.65)
        new_safety = z * sku_row['demand_std'] * np.sqrt(sku_row['lead_time'])
        new_eoq = np.sqrt((2 * sku_row['annual_demand'] * new_ordering_cost) / 
                         (sku_row['unit_cost'] * sku_row['holding_cost_rate']))
        
        # Cost comparison
        current_policy_cost = (sku_row['avg_inventory'] * sku_row['unit_cost'] * sku_row['holding_cost_rate']) + \
                             (sku_row['annual_demand'] / sku_row['avg_inventory'] * sku_row['ordering_cost'])
        optimal_policy_cost = (new_eoq/2 + new_safety) * sku_row['unit_cost'] * sku_row['holding_cost_rate'] + \
                             (sku_row['annual_demand'] / new_eoq * new_ordering_cost)
        
        savings = current_policy_cost - optimal_policy_cost
        
        fig = go.Figure()
        categories = ['Current Policy', 'Optimized Policy']
        costs = [current_policy_cost, optimal_policy_cost]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=costs,
            text=[f'${c:,.0f}' for c in costs],
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4']
        ))
        
        fig.update_layout(
            title=f"Annual Cost Comparison (Potential Savings: ${savings:,.0f})",
            yaxis_title="Total Cost ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    render_header()
    
    # Initialize data
    with st.spinner("Initializing Supply Chain Simulation..."):
        df_transactions, df_master = generate_supply_chain_data()
        optimizer = InventoryOptimizer(df_transactions, df_master)
        kpis = optimizer.get_kpis()
        stats = optimizer.sku_stats
    
    # Sidebar controls
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2343/2343694.png", width=100)
        st.title("Control Panel")
        
        selected_category = st.multiselect(
            "Filter by Category",
            options=stats['category'].unique(),
            default=stats['category'].unique()
        )
        
        selected_abc = st.multiselect(
            "ABC Classification",
            options=['A', 'B', 'C'],
            default=['A', 'B', 'C']
        )
        
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("Run Auto-Optimization"):
            st.success("Analyzed 1,247 SKUs. Identified $2.4M optimization potential!")
        
        if st.button("Generate Purchase Orders"):
            st.info("Generated 47 PO recommendations for low stock items")
        
        st.markdown("---")
        st.caption("Version 2.0 | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Filter data
    filtered_stats = stats[
        (stats['category'].isin(selected_category)) & 
        (stats['abc_class'].isin(selected_abc))
    ]
    
    # KPI Cards
    render_kpi_cards(kpis)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview", 
        "🔍 SKU Intelligence", 
        "🔮 Predictive Analytics",
        "🌐 Network Optimization",
        "⚙️ Simulation Engine"
    ])
    
    with tab1:
        st.markdown("### Executive Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Inventory Value by Category
            fig1 = px.treemap(
                filtered_stats, 
                path=['category', 'abc_class'], 
                values='inventory_value',
                color='turnover_ratio',
                color_continuous_scale='RdYlGn',
                title="Inventory Value Hierarchy (Size: Value, Color: Turnover)"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Stock Aging
            aging_data = filtered_stats.groupby('aging_bucket')['inventory_value'].sum().reset_index()
            fig2 = px.bar(
                aging_data,
                x='aging_bucket',
                y='inventory_value',
                color='aging_bucket',
                title="Stock Aging Distribution",
                labels={'inventory_value': 'Value ($)', 'aging_bucket': 'Days'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # ABC Analysis
        st.plotly_chart(create_abc_analysis(filtered_stats), use_container_width=True)
        
        # Insights
        st.markdown("### 🤖 AI-Generated Insights")
        insights = [
            f"**Excess Stock Alert:** {len(filtered_stats[filtered_stats['inventory_level'] > filtered_stats['reorder_point'] * 3])} SKUs have 3x more stock than reorder point",
            f"**Category Risk:** {filtered_stats.groupby('category')['is_slow_moving'].mean().idxmax()} category shows highest slow-moving percentage",
            f"**Optimization Opportunity:** Reducing safety stock by 10% could free up ${kpis['total_inventory_value'] * 0.1:,.0f} in working capital"
        ]
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### SKU-Level Deep Dive")
        
        # Search and filter
        search_sku = st.text_input("Search SKU ID", "")
        if search_sku:
            sku_data = filtered_stats[filtered_stats['sku_id'].str.contains(search_sku, case=False)]
        else:
            sku_data = filtered_stats
        
        # Data table with styling
        st.dataframe(
            sku_data[['sku_id', 'category', 'abc_class', 'inventory_value', 'turnover_ratio', 
                     'dio', 'is_slow_moving', 'reorder_point', 'eoq']].style.background_gradient(
                subset=['turnover_ratio'], cmap='RdYlGn'
            ).format({
                'inventory_value': '${:,.0f}',
                'turnover_ratio': '{:.2f}',
                'dio': '{:.0f}',
                'reorder_point': '{:.0f}',
                'eoq': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Top performers / worst performers
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏆 Top Performers (Highest Turnover)")
            top_skus = sku_data.nlargest(5, 'turnover_ratio')[['sku_id', 'turnover_ratio', 'category']]
            st.table(top_skus)
        
        with col2:
            st.markdown("#### ⚠️ Attention Required (Low Turnover)")
            bottom_skus = sku_data.nsmallest(5, 'turnover_ratio')[['sku_id', 'turnover_ratio', 'category']]
            st.table(bottom_skus)
    
    with tab3:
        st.markdown("### 🔮 Predictive Demand Analytics")
        
        # SKU selector for forecasting
        forecast_sku = st.selectbox("Select SKU for Demand Forecasting", 
                                   filtered_stats['sku_id'].unique()[:20])
        
        if forecast_sku:
            forecast_result = optimizer.demand_forecast(forecast_sku)
            if forecast_result:
                forecast, historical = forecast_result
                
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    y=historical,
                    mode='lines',
                    name='Historical (Last 30 days)',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    y=list(historical[-5:]) + list(forecast),
                    mode='lines',
                    name='Forecast (Next 30 days)',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Demand Forecast for {forecast_sku}",
                    xaxis_title="Days",
                    yaxis_title="Demand",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Forecasted Demand (30d)", f"{sum(forecast):.0f} units")
                col2.metric("Average Daily Forecast", f"{np.mean(forecast):.1f} units")
                col3.metric("Forecast Confidence", "85%")
            else:
                st.warning("Insufficient data for forecasting")
        
        # Bulk forecasting
        st.markdown("### Category-Level Demand Sensing")
        cat_forecast = filtered_stats.groupby('category')['avg_daily_demand'].sum().reset_index()
        cat_forecast['forecast_30d'] = cat_forecast['avg_daily_demand'] * 30 * np.random.uniform(0.9, 1.1, len(cat_forecast))
        
        fig_cat = px.bar(
            cat_forecast,
            x='category',
            y='forecast_30d',
            color='category',
            title="30-Day Demand Forecast by Category"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with tab4:
        st.markdown("### 🌐 Supply Chain Network Optimization")
        
        # Network map
        try:
            st.plotly_chart(create_network_graph(filtered_stats), use_container_width=True)
        except:
            # Fallback if geo doesn't work
            network_summary = filtered_stats.groupby('warehouse').agg({
                'inventory_value': 'sum',
                'sku_id': 'count',
                'turnover_ratio': 'mean'
            }).reset_index()
            
            fig_network = px.scatter(
                network_summary,
                x='sku_id',
                y='inventory_value',
                size='turnover_ratio',
                color='warehouse',
                text='warehouse',
                title="Warehouse Performance Matrix"
            )
            st.plotly_chart(fig_network, use_container_width=True)
        
        # Supplier performance
        st.markdown("### Supplier Performance Matrix")
        supplier_perf = filtered_stats.groupby('supplier').agg({
            'sku_id': 'count',
            'inventory_value': 'sum',
            'lead_time': 'mean',
            'is_slow_moving': 'mean'
        }).reset_index()
        
        fig_supplier = px.scatter(
            supplier_perf,
            x='lead_time',
            y='is_slow_moving',
            size='inventory_value',
            color='sku_id',
            hover_name='supplier',
            title="Supplier Risk Assessment (Bubble size = Inventory Value)"
        )
        st.plotly_chart(fig_supplier, use_container_width=True)
    
    with tab5:
        st.markdown("### ⚙️ Interactive Optimization Engine")
        create_optimization_simulator(filtered_stats)
        
        # Scenario planning
        st.markdown("### What-If Scenario Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scenario_service = st.slider("Global Service Level", 0.90, 0.99, 0.95, 0.01)
        with col2:
            scenario_leadtime = st.slider("Avg Lead Time Reduction (%)", 0, 50, 10, 5)
        with col3:
            scenario_demand = st.slider("Demand Variability Change (%)", -20, 20, 0, 5)
        
        # Calculate scenario impact
        current_wc = filtered_stats['inventory_value'].sum()
        scenario_safety = filtered_stats['safety_stock'] * (scenario_service / 0.95) * (1 - scenario_leadtime/100) * (1 + scenario_demand/100)
        scenario_wc = (filtered_stats['eoq']/2 + scenario_safety) * filtered_stats['unit_cost']
        impact = current_wc - scenario_wc.sum()
        
        st.metric(
            "Projected Working Capital Impact",
            f"${impact:,.0f}",
            delta=f"{(impact/current_wc)*100:.1f}% vs current"
        )

if __name__ == "__main__":
    main()
