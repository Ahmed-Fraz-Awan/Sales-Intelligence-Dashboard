# ULTIMATE SALES INTELLIGENCE DASHBOARD - MERGED VERSION
# Install: pip install dash plotly pandas numpy scikit-learn dash-bootstrap-components

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============= DATA GENERATION =============
def create_advanced_data():
    """Create comprehensive e-commerce data with ML-ready features"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    categories = ['Furniture', 'Technology', 'Office Supplies']
    regions = ['East', 'West', 'Central', 'South']
    segments = ['Consumer', 'Corporate', 'Home Office']
    
    data = []
    customer_id = 1000
    
    for date in dates:
        month = date.month
        seasonal = 1.0
        if month in [11, 12]:
            seasonal = 1.8
        elif month in [6, 7, 8]:
            seasonal = 0.7
        
        for _ in range(np.random.randint(3, 8)):
            category = np.random.choice(categories, p=[0.3, 0.45, 0.25])
            base_sales = {
                'Furniture': np.random.uniform(200, 3000),
                'Technology': np.random.uniform(300, 5000),
                'Office Supplies': np.random.uniform(50, 800)
            }[category]
            
            sales = base_sales * seasonal * np.random.uniform(0.8, 1.2)
            profit_margin = {
                'Furniture': 0.12,
                'Technology': 0.15,
                'Office Supplies': 0.25
            }[category]
            
            data.append({
                'Order_Date': date,
                'Customer_ID': f'CUST-{customer_id + np.random.randint(1, 500)}',
                'Category': category,
                'Region': np.random.choice(regions, p=[0.28, 0.32, 0.22, 0.18]),
                'Segment': np.random.choice(segments, p=[0.5, 0.3, 0.2]),
                'Sales': sales,
                'Profit': sales * profit_margin * np.random.uniform(0.7, 1.3),
                'Quantity': np.random.randint(1, 15),
                'Discount': np.random.choice([0, 0.1, 0.15, 0.2], p=[0.6, 0.2, 0.15, 0.05])
            })
    
    df = pd.DataFrame(data)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Quarter'] = df['Order_Date'].dt.quarter
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Profit_Margin'] = (df['Profit'] / df['Sales'] * 100)
    
    return df

df = create_advanced_data()

# ============= ML MODELS =============
def train_forecasting_model(data):
    """Train Random Forest for sales forecasting"""
    monthly_sales = data.groupby(pd.Grouper(key='Order_Date', freq='M'))['Sales'].sum().reset_index()
    monthly_sales['Month_Num'] = range(len(monthly_sales))
    monthly_sales['Month'] = monthly_sales['Order_Date'].dt.month
    monthly_sales['Year'] = monthly_sales['Order_Date'].dt.year
    
    for lag in [1, 2, 3]:
        monthly_sales[f'Sales_Lag_{lag}'] = monthly_sales['Sales'].shift(lag)
    
    monthly_sales = monthly_sales.dropna()
    
    X = monthly_sales[['Month_Num', 'Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3']]
    y = monthly_sales['Sales']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_date = monthly_sales['Order_Date'].max()
    last_month_num = monthly_sales['Month_Num'].max()
    last_sales = monthly_sales['Sales'].tail(3).values
    
    forecasts = []
    for i in range(1, 7):
        next_date = last_date + pd.DateOffset(months=i)
        next_features = np.array([[
            last_month_num + i,
            next_date.month,
            next_date.year,
            last_sales[2],
            last_sales[1],
            last_sales[0]
        ]])
        
        pred = model.predict(next_features)[0]
        forecasts.append({
            'Date': next_date,
            'Forecast': pred,
            'Lower_Bound': pred * 0.85,
            'Upper_Bound': pred * 1.15
        })
        last_sales = np.append(last_sales[1:], pred)
    
    return pd.DataFrame(forecasts), monthly_sales

forecast_df, historical_monthly = train_forecasting_model(df)

def perform_customer_segmentation(data):
    """RFM Analysis with K-Means clustering"""
    max_date = data['Order_Date'].max()
    rfm = data.groupby('Customer_ID').agg({
        'Order_Date': lambda x: (max_date - x.max()).days,
        'Sales': ['count', 'sum']
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
    
    segment_mapping = {}
    for seg in rfm['Segment'].unique():
        seg_data = rfm[rfm['Segment'] == seg]
        avg_recency = seg_data['Recency'].mean()
        avg_frequency = seg_data['Frequency'].mean()
        
        if avg_recency < 100 and avg_frequency > 5:
            segment_mapping[seg] = 'Champions'
        elif avg_recency < 180 and avg_frequency > 3:
            segment_mapping[seg] = 'Loyal'
        elif avg_recency < 365:
            segment_mapping[seg] = 'At Risk'
        else:
            segment_mapping[seg] = 'Lost'
    
    rfm['Segment_Name'] = rfm['Segment'].map(segment_mapping)
    return rfm

customer_segments = perform_customer_segmentation(df)

# ============= DASH APP =============
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True)
app.title = "Ultimate Sales Intelligence Dashboard"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            body {
                font-family: 'Inter', sans-serif;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            
            .gradient-text {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            }
            
            /* Dropdown z-index fix */
            .Select-menu-outer {
                z-index: 1000 !important;
            }
            
            .DateInput_input {
                z-index: 1000 !important;
            }
            
            .DateRangePicker_picker {
                z-index: 1001 !important;
            }
            
            /* Ensure KPI cards don't overlap dropdowns */
            .kpi-section {
                position: relative;
                z-index: 1;
            }
            
            .kpi-card {
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.2) !important;
            }
            
            .tab-content {
                animation: fadeIn 0.5s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============= LAYOUT =============
app.layout = dbc.Container([
    # Header with gradient background
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    "Ultimate Sales Intelligence Dashboard"
                ], className="display-3 fw-bold text-center mb-2", 
                   style={'color': 'white', 'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'}),
                html.P("AI-Powered Analytics • ML Forecasting • Business Intelligence",
                      className="lead text-center mb-4",
                      style={'color': 'rgba(255,255,255,0.9)', 'fontSize': '1.3rem'})
            ], style={'padding': '40px 0'})
        ])
    ]),
    
    # Control Panel with proper z-index
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label([html.I(className="fas fa-calendar-alt me-2"), "Date Range"], 
                              className="fw-bold mb-2", style={'fontSize': '14px'}),
                    dcc.DatePickerRange(
                        id='date-filter',
                        start_date=df['Order_Date'].min(),
                        end_date=df['Order_Date'].max(),
                        display_format='MMM DD, YYYY',
                        style={'width': '100%', 'zIndex': '1000'}
                    ),
                ], md=4),
                dbc.Col([
                    html.Label([html.I(className="fas fa-box me-2"), "Category Filter"], 
                              className="fw-bold mb-2", style={'fontSize': '14px'}),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': '🌟 All Categories', 'value': 'All'}] + 
                               [{'label': f'📦 {cat}', 'value': cat} for cat in df['Category'].unique()],
                        value='All',
                        clearable=False,
                        style={'zIndex': '1000'}
                    ),
                ], md=4),
                dbc.Col([
                    html.Label([html.I(className="fas fa-map-marker-alt me-2"), "Region Filter"], 
                              className="fw-bold mb-2", style={'fontSize': '14px'}),
                    dcc.Dropdown(
                        id='region-filter',
                        options=[{'label': '🌟 All Regions', 'value': 'All'}] + 
                               [{'label': f'📍 {reg}', 'value': reg} for reg in df['Region'].unique()],
                        value='All',
                        clearable=False,
                        style={'zIndex': '1000'}
                    ),
                ], md=4),
            ])
        ])
    ], className="glass-card shadow mb-4", style={'position': 'relative', 'zIndex': '999'}),
    
    # KPI Cards Section with proper z-index
    html.Div(id='kpi-section', className="mb-4 kpi-section"),
    
    # Main Dashboard with Tabs
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs([
                # Tab 1: Overview & Forecasting
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='forecast-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm h-100")
                            ], md=8),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='category-pie', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm mb-3"),
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='funnel-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=4),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='trend-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=12),
                        ])
                    ], className="p-3")
                ], label="📈 Overview & Forecasting", tab_id="tab-1", 
                   label_style={'fontSize': '16px', 'fontWeight': '600'}),
                
                # Tab 2: Performance Analysis
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='heatmap-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='segment-sunburst', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='regional-bar', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='category-treemap', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                        ])
                    ], className="p-3")
                ], label="🎯 Performance Analysis", tab_id="tab-2",
                   label_style={'fontSize': '16px', 'fontWeight': '600'}),
                
                # Tab 3: Customer Intelligence
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='customer-segments', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='scatter-analysis', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='profit-margin-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id='waterfall-chart', config={'displayModeBar': False})
                                    ])
                                ], className="shadow-sm")
                            ], md=6),
                        ])
                    ], className="p-3")
                ], label="👥 Customer Intelligence", tab_id="tab-3",
                   label_style={'fontSize': '16px', 'fontWeight': '600'}),
                
                # Tab 4: AI Insights
                dbc.Tab([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4([
                                    html.I(className="fas fa-lightbulb me-2"),
                                    "AI-Generated Insights & Recommendations"
                                ], className="mb-0 text-primary")
                            ]),
                            dbc.CardBody([
                                html.Div(id='insights-content')
                            ])
                        ], className="shadow-sm")
                    ], className="p-3")
                ], label="💡 AI Insights", tab_id="tab-4",
                   label_style={'fontSize': '16px', 'fontWeight': '600'}),
            ], id="tabs", active_tab="tab-1")
        ])
    ], className="glass-card shadow-lg"),
    
    # Footer
    html.Div([
        html.Hr(style={'borderTop': '2px solid rgba(255,255,255,0.3)', 'margin': '40px 0 20px 0'}),
        html.P([
            html.I(className="fas fa-copyright me-2"),
            "2024 Ultimate Sales Intelligence Dashboard | Powered by Machine Learning & Advanced Analytics"
        ], className="text-center small", 
           style={'color': 'rgba(255,255,255,0.8)', 'fontSize': '14px'})
    ])
    
], fluid=True, style={'minHeight': '100vh', 'padding': '20px'})

# ============= CALLBACKS =============
@app.callback(
    [Output('kpi-section', 'children'),
     Output('forecast-chart', 'figure'),
     Output('category-pie', 'figure'),
     Output('trend-chart', 'figure'),
     Output('heatmap-chart', 'figure'),
     Output('segment-sunburst', 'figure'),
     Output('regional-bar', 'figure'),
     Output('category-treemap', 'figure'),
     Output('customer-segments', 'figure'),
     Output('scatter-analysis', 'figure'),
     Output('profit-margin-chart', 'figure'),
     Output('waterfall-chart', 'figure'),
     Output('funnel-chart', 'figure'),
     Output('insights-content', 'children')],
    [Input('date-filter', 'start_date'),
     Input('date-filter', 'end_date'),
     Input('category-filter', 'value'),
     Input('region-filter', 'value')]
)
def update_dashboard(start_date, end_date, category, region):
    # Filter data
    filtered = df.copy()
    if start_date and end_date:
        filtered = filtered[(filtered['Order_Date'] >= start_date) & 
                          (filtered['Order_Date'] <= end_date)]
    if category != 'All':
        filtered = filtered[filtered['Category'] == category]
    if region != 'All':
        filtered = filtered[filtered['Region'] == region]
    
    # Calculate KPIs
    total_sales = filtered['Sales'].sum()
    total_profit = filtered['Profit'].sum()
    total_orders = len(filtered)
    avg_order = total_sales / total_orders if total_orders > 0 else 0
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    unique_customers = filtered['Customer_ID'].nunique()
    
    # Calculate growth
    if len(filtered) > 30:
        recent = filtered.tail(30)['Sales'].sum()
        previous = filtered.head(30)['Sales'].sum()
        growth = ((recent - previous) / previous * 100) if previous > 0 else 0
    else:
        growth = 0
    
    # KPI Cards with gradient backgrounds
    kpi_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Total Revenue", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"${total_sales:,.0f}", className="mb-1 fw-bold"),
                        html.Small([
                            html.I(className="fas fa-arrow-up me-1" if growth > 0 else "fas fa-arrow-down me-1"),
                            f"{abs(growth):.1f}% vs previous"
                        ], className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card", 
               style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Net Profit", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"${total_profit:,.0f}", className="mb-1 fw-bold"),
                        html.Small(f"{profit_margin:.1f}% margin", className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card",
               style={'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-shopping-cart fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Total Orders", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"{total_orders:,}", className="mb-1 fw-bold"),
                        html.Small(f"Avg: ${avg_order:,.0f}", className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card",
               style={'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-credit-card fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Avg Order Value", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"${avg_order:,.0f}", className="mb-1 fw-bold"),
                        html.Small("Per transaction", className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card",
               style={'background': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-percentage fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Profit Margin", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"{profit_margin:.1f}%", className="mb-1 fw-bold"),
                        html.Small("Overall margin", className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card",
               style={'background': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-users fa-2x mb-2", style={'opacity': '0.8'}),
                        html.H6("Customers", className="text-white-50 mb-2 fw-normal"),
                        html.H2(f"{unique_customers:,}", className="mb-1 fw-bold"),
                        html.Small("Unique buyers", className="text-white-50")
                    ])
                ])
            ], className="shadow kpi-card",
               style={'background': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)', 
                      'color': 'white', 'borderRadius': '15px'})
        ], md=2),
    ])
    
    # 1. ML Forecast Chart
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=historical_monthly['Order_Date'], y=historical_monthly['Sales'],
        mode='lines+markers', name='Historical Sales',
        line=dict(color='#667eea', width=3),
        fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)',
        marker=dict(size=6)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'], y=forecast_df['Forecast'],
        mode='lines+markers', name='AI Forecast',
        line=dict(color='#f093fb', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
        fill='toself', fillcolor='rgba(240, 147, 251, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval', showlegend=True
    ))
    fig_forecast.update_layout(
        title={'text': '<b>🤖 AI-Powered Sales Forecast (Random Forest Model)</b>',
               'font': {'size': 18}},
        template='plotly_white', height=380,
        hovermode='x unified', showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # 2. Category Pie Chart
    cat_data = filtered.groupby('Category')['Sales'].sum().reset_index()
    fig_pie = px.pie(cat_data, values='Sales', names='Category',
                     title='<b>📊 Sales Distribution by Category</b>',
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     hole=0.4)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                          textfont_size=12)
    fig_pie.update_layout(height=350, showlegend=True)
    
    # 3. Monthly Trend with Dual Axis
    monthly_trend = filtered.groupby(pd.Grouper(key='Order_Date', freq='M')).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly_trend['Order_Date'], y=monthly_trend['Sales'],
               name='Sales', marker_color='rgba(102, 126, 234, 0.7)'),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly_trend['Order_Date'], y=monthly_trend['Profit'],
                   name='Profit', line=dict(color='#f093fb', width=3),
                   mode='lines+markers', marker=dict(size=8)),
        secondary_y=True
    )
    fig_trend.update_layout(
        title={'text': '<b>📈 Monthly Sales & Profit Trend Analysis</b>',
               'font': {'size': 18}},
        template='plotly_white', height=400,
        hovermode='x unified'
    )
    fig_trend.update_yaxes(title_text="<b>Sales ($)</b>", secondary_y=False)
    fig_trend.update_yaxes(title_text="<b>Profit ($)</b>", secondary_y=True)
    
    # 4. Sales Heatmap
    heatmap_data = filtered.groupby(['Month', 'Day_of_Week'])['Sales'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Day_of_Week', columns='Month', values='Sales')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='Viridis',
        text=heatmap_pivot.values,
        texttemplate='$%{text:.0f}',
        textfont={"size": 10},
        colorbar=dict(title="Avg Sales ($)")
    ))
    fig_heatmap.update_layout(
        title={'text': '<b>🔥 Sales Heatmap: Day of Week vs Month</b>',
               'font': {'size': 18}},
        height=400,
        template='plotly_white'
    )
    
    # 5. Sunburst - Multi-level Hierarchy
    segment_data = filtered.groupby(['Region', 'Category', 'Segment'])['Sales'].sum().reset_index()
    fig_sunburst = px.sunburst(
        segment_data,
        path=['Region', 'Category', 'Segment'],
        values='Sales',
        color='Sales',
        color_continuous_scale='plasma',
        title='<b>🌟 Multi-Level Sales Hierarchy</b>'
    )
    fig_sunburst.update_layout(height=400, font=dict(size=12))
    
    # 6. Regional Bar Chart
    region_data = filtered.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index().sort_values('Sales', ascending=True)
    
    fig_regional = go.Figure()
    fig_regional.add_trace(go.Bar(
        y=region_data['Region'], 
        x=region_data['Sales'],
        name='Sales', 
        orientation='h',
        marker=dict(
            color=region_data['Sales'],
            colorscale='Viridis',
            showscale=True
        ),
        text=region_data['Sales'].apply(lambda x: f'${x:,.0f}'),
        textposition='auto'
    ))
    fig_regional.update_layout(
        title={'text': '<b>🗺️ Regional Sales Performance</b>',
               'font': {'size': 18}},
        template='plotly_white', 
        height=400,
        xaxis_title='Sales ($)',
        yaxis_title='Region'
    )
    
    # 7. Category Treemap
    treemap_data = filtered.groupby(['Category', 'Segment']).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    fig_treemap = px.treemap(
        treemap_data,
        path=['Category', 'Segment'],
        values='Sales',
        color='Profit',
        color_continuous_scale='RdYlGn',
        title='<b>📦 Sales & Profitability Treemap</b>'
    )
    fig_treemap.update_layout(height=400)
    fig_treemap.update_traces(textinfo="label+value+percent parent")
    
    # 8. Customer Segmentation Analysis
    segment_summary = customer_segments.groupby('Segment_Name').agg({
        'Customer_ID': 'count',
        'Monetary': 'mean',
        'Frequency': 'mean'
    }).reset_index()
    
    fig_segments = go.Figure()
    colors = {
        'Champions': '#43e97b', 
        'Loyal': '#4facfe', 
        'At Risk': '#fa709a', 
        'Lost': '#667eea'
    }
    
    for segment in segment_summary['Segment_Name']:
        data = segment_summary[segment_summary['Segment_Name'] == segment]
        fig_segments.add_trace(go.Bar(
            name=segment,
            x=['Customers', 'Avg Revenue ($)', 'Avg Orders'],
            y=[data['Customer_ID'].values[0], 
               data['Monetary'].values[0], 
               data['Frequency'].values[0]],
            marker_color=colors.get(segment, '#667eea'),
            text=[f"{data['Customer_ID'].values[0]:.0f}",
                  f"${data['Monetary'].values[0]:.0f}",
                  f"{data['Frequency'].values[0]:.1f}"],
            textposition='auto'
        ))
    
    fig_segments.update_layout(
        title={'text': '<b>👥 Customer Segmentation (RFM Analysis)</b>',
               'font': {'size': 18}},
        barmode='group', 
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    # 9. Scatter Analysis - Profitability Matrix
    scatter_data = filtered.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    fig_scatter = px.scatter(
        scatter_data, 
        x='Sales', 
        y='Profit',
        size='Quantity', 
        color='Category',
        text='Category',
        title='<b>💎 Profitability Matrix: Sales vs Profit</b>',
        color_discrete_sequence=px.colors.qualitative.Bold,
        size_max=60
    )
    fig_scatter.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=2, color='white'))
    )
    fig_scatter.update_layout(
        height=400, 
        template='plotly_white',
        xaxis_title='Total Sales ($)',
        yaxis_title='Total Profit ($)'
    )
    
    # 10. Profit Margin by Category
    margin_data = filtered.groupby('Category')['Profit_Margin'].mean().reset_index()
    margin_data = margin_data.sort_values('Profit_Margin', ascending=False)
    
    fig_margin = px.bar(
        margin_data, 
        x='Category', 
        y='Profit_Margin',
        title='<b>📊 Average Profit Margin by Category</b>',
        color='Profit_Margin',
        color_continuous_scale='RdYlGn',
        text=margin_data['Profit_Margin'].apply(lambda x: f'{x:.1f}%')
    )
    fig_margin.update_traces(textposition='outside')
    fig_margin.update_layout(
        height=400, 
        template='plotly_white', 
        showlegend=False,
        yaxis_title='Profit Margin (%)',
        xaxis_title='Category'
    )
    
    # 11. Waterfall Chart - Regional Contribution
    region_sales = filtered.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Regional Sales",
        orientation="v",
        measure=['relative'] * len(region_sales) + ['total'],
        x=region_sales['Region'].tolist() + ['Total'],
        y=region_sales['Sales'].tolist() + [region_sales['Sales'].sum()],
        text=[f"${v:,.0f}" for v in region_sales['Sales']] + [f"${region_sales['Sales'].sum():,.0f}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#fa709a"}},
        increasing={"marker": {"color": "#43e97b"}},
        totals={"marker": {"color": "#667eea"}}
    ))
    
    fig_waterfall.update_layout(
        title={'text': '<b>💧 Regional Sales Contribution Waterfall</b>',
               'font': {'size': 18}},
        height=400,
        template='plotly_white',
        showlegend=False,
        yaxis_title='Sales ($)'
    )
    
    # 12. Funnel Chart - Quarterly Performance
    funnel_data = filtered.groupby('Quarter')['Sales'].sum().reset_index()
    funnel_data = funnel_data.sort_values('Sales', ascending=False)
    
    fig_funnel = go.Figure(go.Funnel(
        y=['Q' + str(q) for q in funnel_data['Quarter']],
        x=funnel_data['Sales'],
        textinfo="value+percent initial",
        marker=dict(
            color=["#667eea", "#764ba2", "#f093fb", "#f5576c"][:len(funnel_data)]
        ),
        textfont=dict(size=14, color='white')
    ))
    
    fig_funnel.update_layout(
        title={'text': '<b>🎯 Quarterly Sales Funnel</b>',
               'font': {'size': 18}},
        height=350,
        template='plotly_white'
    )
    
    # 13. AI-Generated Insights
    top_category = filtered.groupby('Category')['Sales'].sum().idxmax()
    top_category_sales = filtered[filtered['Category']==top_category]['Sales'].sum()
    top_region = filtered.groupby('Region')['Sales'].sum().idxmax()
    top_region_sales = filtered[filtered['Region']==top_region]['Sales'].sum()
    best_segment = filtered.groupby('Segment')['Sales'].sum().idxmax()
    best_segment_sales = filtered[filtered['Segment']==best_segment]['Sales'].sum()
    
    # Calculate best performing month
    monthly_perf = filtered.groupby(filtered['Order_Date'].dt.to_period('M'))['Sales'].sum()
    if len(monthly_perf) > 0:
        best_month = monthly_perf.idxmax()
        best_month_sales = monthly_perf.max()
    else:
        best_month = "N/A"
        best_month_sales = 0
    
    # Customer insights
    avg_customer_value = filtered.groupby('Customer_ID')['Sales'].sum().mean()
    top_customers = filtered.groupby('Customer_ID')['Sales'].sum().nlargest(10).sum()
    top_customers_pct = (top_customers / total_sales * 100) if total_sales > 0 else 0
    
    insights = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-trophy fa-3x mb-3", style={'color': '#667eea'}),
                        html.H4("🎯 Top Performing Category", className="mb-3 fw-bold"),
                        html.P([
                            html.Strong(top_category, style={'fontSize': '20px', 'color': '#667eea'}),
                            " leads with ",
                            html.Strong(f"${top_category_sales:,.0f}", style={'fontSize': '20px', 'color': '#43e97b'}),
                            " in sales, accounting for ",
                            html.Strong(f"{(top_category_sales/total_sales*100):.1f}%", style={'color': '#fa709a'}),
                            " of total revenue."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-map-marked-alt fa-3x mb-3", style={'color': '#4facfe'}),
                        html.H4("🌍 Strongest Region", className="mb-3 fw-bold"),
                        html.P([
                            html.Strong(top_region, style={'fontSize': '20px', 'color': '#4facfe'}),
                            " region dominates with ",
                            html.Strong(f"${top_region_sales:,.0f}", style={'fontSize': '20px', 'color': '#43e97b'}),
                            " in revenue, representing ",
                            html.Strong(f"{(top_region_sales/total_sales*100):.1f}%", style={'color': '#fa709a'}),
                            " of total sales."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-users fa-3x mb-3", style={'color': '#f093fb'}),
                        html.H4("👥 Key Customer Segment", className="mb-3 fw-bold"),
                        html.P([
                            html.Strong(best_segment, style={'fontSize': '20px', 'color': '#f093fb'}),
                            " segment contributes ",
                            html.Strong(f"${best_segment_sales:,.0f}", style={'fontSize': '20px', 'color': '#43e97b'}),
                            ", with an average customer value of ",
                            html.Strong(f"${avg_customer_value:,.0f}", style={'color': '#667eea'}),
                            "."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-3x mb-3", style={'color': '#43e97b'}),
                        html.H4("📊 Profitability Insight", className="mb-3 fw-bold"),
                        html.P([
                            "Overall profit margin is ",
                            html.Strong(f"{profit_margin:.1f}%", style={'fontSize': '20px', 'color': '#43e97b'}),
                            ". Top 10 customers generate ",
                            html.Strong(f"{top_customers_pct:.1f}%", style={'fontSize': '20px', 'color': '#667eea'}),
                            " of revenue, indicating strong customer concentration."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-calendar-check fa-3x mb-3", style={'color': '#764ba2'}),
                        html.H4("📅 Best Month Performance", className="mb-3 fw-bold"),
                        html.P([
                            "Peak sales month was ",
                            html.Strong(str(best_month), style={'fontSize': '20px', 'color': '#764ba2'}),
                            " with ",
                            html.Strong(f"${best_month_sales:,.0f}", style={'fontSize': '20px', 'color': '#43e97b'}),
                            " in revenue, indicating strong seasonal patterns."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-robot fa-3x mb-3", style={'color': '#f5576c'}),
                        html.H4("🔮 ML Forecast Prediction", className="mb-3 fw-bold"),
                        html.P([
                            "Next 6 months projected revenue: ",
                            html.Strong(f"${forecast_df['Forecast'].sum():,.0f}", 
                                      style={'fontSize': '20px', 'color': '#f5576c'}),
                            ". Confidence interval ranges from ",
                            html.Strong(f"${forecast_df['Lower_Bound'].sum():,.0f}", 
                                      style={'color': '#fa709a'}),
                            " to ",
                            html.Strong(f"${forecast_df['Upper_Bound'].sum():,.0f}", 
                                      style={'color': '#43e97b'}),
                            "."
                        ], className="mb-0")
                    ])
                ])
            ], className="shadow-sm mb-3")
        ], md=6),
    ])
    
    return (kpi_cards, fig_forecast, fig_pie, fig_trend, fig_heatmap, 
            fig_sunburst, fig_regional, fig_treemap, fig_segments, 
            fig_scatter, fig_margin, fig_waterfall, fig_funnel, insights)


server = app.server  # Expose the server variable for deployment

if __name__ == '__main__':
   #app.run(debug=True, port=8050)
#if __name__ == '__main__':
    print("🚀 Starting Ultimate Sales Intelligence Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("✨ Features: ML Forecasting | Customer Segmentation | Real-time Analytics")
    app.run(debug=True, port=8050)