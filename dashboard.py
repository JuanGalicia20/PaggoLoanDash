import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from lib.lib import consulta

# Colores de marca
PRIMARY_COLOR = "#4ACDCE"
SECONDARY_COLOR = "#E044A7"
TERTIARY_COLOR = "#5266B0"


# ——— Login ———
def login():
    st.title("Iniciar sesión")
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Entrar")

    if login_button:
        if (
            email in st.secrets["usuarios"]
            and password == st.secrets["usuarios"][email]
        ):
            st.session_state["logueado"] = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas. Inténtalo de nuevo.")

# Inicializar estado de sesión
if "logueado" not in st.session_state:
    st.session_state["logueado"] = False

# Asegurar autenticación antes de continuar
if not st.session_state["logueado"]:
    login()
    st.stop()

# Configuración de página
st.set_page_config(
    page_title="Simulador Avanzado de Préstamos Paggo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados con colores de marca
st.markdown(f"""
<style>
    .main-header {{ font-size: 2.3rem; font-weight: 700; color: {PRIMARY_COLOR}; margin-bottom: 1rem; }}
    .subheader {{ font-size: 1.5rem; font-weight: 600; color: {SECONDARY_COLOR}; margin-top: 1rem; margin-bottom: 0.5rem; }}
    .card {{ background-color: #F3F4F6; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); }}
    .metric-title {{ font-size: 1rem; font-weight: 500; color: #6B7280; }}
    .metric-value {{ font-size: 1.8rem; font-weight: 700; color: {PRIMARY_COLOR}; }}
    .metric-change {{ font-size: 0.8rem; font-weight: 500; }}
    .positive-change {{ color: #10B981; }}
    .negative-change {{ color: #EF4444; }}
    .neutral-change {{ color: {TERTIARY_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown(f'<div class="main-header">Simulador Avanzado de Préstamos de Capital de Trabajo</div>', unsafe_allow_html=True)

# Cargar datos de riesgo MCC
@st.cache_data(ttl=3600)
def load_mcc_risk_data():
    """
    Carga los datos de riesgo por MCC desde CSV o base de datos
    """
    # Opción 1: Si tienes el CSV subido
    try:
        # Si el CSV está en el sistema de archivos
        mcc_risk = pd.read_csv('mcc_risk.csv')
        return mcc_risk
    except:
        # Opción 2: Datos hardcodeados como fallback
        mcc_risk_data = {
            'MCC': [7230, 7994, 5532],
            'Descripción': ['Sala de belleza', 'Videojuegos', 'Venta de neumáticos'],
            'Probabilidad': ['Bajo', 'Bajo', 'Bajo'],
            'Consecuencia': ['Menor', 'Moderado', 'Mayor'],
            'Calificación': [2, 4, 8],
            'Nivel de riesgo': ['', '', '']
        }
        return pd.DataFrame(mcc_risk_data)


# Cargar datos (cache por 1 hora)
@st.cache_data(ttl=3600)
def load_transaction_data():
    return consulta(st.secrets["query_negocios"]["query1"])

with st.spinner('Cargando datos de transacciones...'):
    df = load_transaction_data()
    df_mcc_risk = load_mcc_risk_data()
if df.empty:
    st.error("No se pudieron cargar los datos o no hay información disponible.")
    st.stop()

df['codigoMcc'] = df['codigoMcc'].astype(int)
df_mcc_risk['MCC'] = df_mcc_risk['MCC'].astype(int)

# Combinar con datos de riesgo MCC
df = df.merge(
    df_mcc_risk[['MCC', 'Calificación', 'Nivel de riesgo', 'Probabilidad', 'Consecuencia']], 
    left_on='codigoMcc', 
    right_on='MCC', 
    how='left'
)
print(df.head())

# Llenar valores faltantes con riesgo medio por defecto
df['Calificación'] = df['Calificación'].fillna(5)  # Riesgo medio por defecto
df['Nivel de riesgo'] = df['Nivel de riesgo'].fillna('Medio')
df['Probabilidad'] = df['Probabilidad'].fillna('Medio')
df['Consecuencia'] = df['Consecuencia'].fillna('Moderado')
# Preprocesar fechas
df['fechaAfiliacion'] = pd.to_datetime(df['fechaAfiliacion'])


# Sidebar: Configuración del préstamo
st.sidebar.markdown('<div class="subheader">Configuración del Préstamo</div>', unsafe_allow_html=True)
with st.sidebar.container():
    loan_amount = st.slider("Monto del préstamo (Q)", 8000, 12000, 10000, step=1000)
    term_months = st.selectbox("Plazo (meses)", [3, 6, 9, 12], index=1)
    interest_rate = st.slider("Tasa de interés anual (%)", 12.0, 60.0, 36.0, step=1.0) / 100.0
    monthly_interest = interest_rate / 12.0
    repayment_type = st.radio("Tipo de cuota", ["Fija", "Decreciente"])

    if repayment_type == "Fija":
        # Cuota fija según fórmula estándar
        monthly_payment = loan_amount * (monthly_interest * (1 + monthly_interest)**term_months) / ((1 + monthly_interest)**term_months - 1)
    else:
        # Cuota decreciente: comenzando con la más alta
        principal_payment = loan_amount / term_months
        payments_schedule = []
        remaining_balance = loan_amount
        for _ in range(term_months):
            interest_payment = remaining_balance * monthly_interest
            payment_m = principal_payment + interest_payment
            payments_schedule.append(payment_m)
            remaining_balance -= principal_payment
        monthly_payment = payments_schedule[0]

    st.info(f"Cuota mensual {'inicial ' if repayment_type=='Decreciente' else ''}equivalente: Q {monthly_payment:.2f}")

# Sidebar: Configuración de Retención
st.sidebar.markdown('<div class="subheader">Configuración de Retención</div>', unsafe_allow_html=True)
with st.sidebar.container():
    # Retención fija independientemente del tipo de cuota
    retention_rate = st.slider("% Retención sobre ventas", 1, 30, 10) / 100.0
    retention_schedule = np.repeat(retention_rate, term_months)
    platform_share = st.slider("% de ventas que pasan por la plataforma", 50, 100, 70) / 100.0

# Sidebar: Supuestos de Riesgo
st.sidebar.markdown('<div class="subheader">Supuestos de Riesgo</div>', unsafe_allow_html=True)
with st.sidebar.container():
    PD = st.slider("PD - Probabilidad de Default (%)", 0, 30, 10) / 100.0
    st.caption("Probabilidad de que un negocio no cumpla con sus pagos y entre en mora.")
    st.caption("Probabilidad base de default antes de ajustes por MCC.")
    
    # Multiplicadores por nivel de riesgo MCC
    st.subheader("Multiplicadores de Riesgo por MCC")
    
    # Crear multiplicadores basados en la calificación de riesgo
    risk_multiplier_low = st.slider("Multiplicador Riesgo Bajo (Calif. 1-3)", 0.5, 1.5, 0.8, 0.1)
    risk_multiplier_medium = st.slider("Multiplicador Riesgo Medio (Calif. 4-6)", 0.8, 2.0, 1.0, 0.1)
    risk_multiplier_high = st.slider("Multiplicador Riesgo Alto (Calif. 7-10)", 1.0, 3.0, 1.5, 0.1)
    

    LGD = st.slider("LGD - Pérdida dada Default (%)", 0, 100, 90) / 100.0
    st.caption("Porcentaje de la exposición que se pierde cuando ocurre un default.")

    cost_of_funds = st.slider("Coste de fondeo anual (%)", 0, 20, 8) / 100.0
    st.caption("Tasa de interés que pagamos por los fondos que prestamos.")

    operational_cost = st.slider("Gastos operativos por préstamo (Q)", 0, 500, 150)
    st.caption("Costos administrativos y operativos asociados a originar y gestionar el préstamo.")

    min_roi = st.slider("ROI mínimo esperado (%)", 1, 50, 15) / 100.0
    st.caption("Retorno mínimo que esperamos obtener sobre el capital prestado.")


st.sidebar.markdown('<div class="subheader">Criterios de Calificación</div>', unsafe_allow_html=True)

# Criterios de calificación adicionales
criteria_container = st.sidebar.container()
with criteria_container:
    min_months_active = st.slider("Meses activos mínimos (de los últimos 6)", 1, 6, 6)
    min_weeks_active = st.slider("Semanas activas mínimas (de las últimas 26)", 4, 26, 20)
    min_antiguedad = st.slider("Antigüedad mínima (meses)", 6, 24, 6)

# Cálculo umbral de facturación
total_retention_avg = sum(retention_schedule) / len(retention_schedule)
monthly_payment_total = monthly_payment * term_months
required_total_retention = monthly_payment_total / (total_retention_avg * platform_share)
monthly_threshold = required_total_retention / term_months

st.sidebar.markdown(f"""
<div class="card">
    <div class="metric-title">Umbral ventas mensuales</div>
    <div class="metric-value">Q {monthly_threshold:,.0f}</div>
    <div class="metric-change neutral-change">Para asegurar pago con {total_retention_avg*100:.1f}% de retención</div>
</div>
""", unsafe_allow_html=True)

# Calculadora de amortización
def generate_amortization_schedule(loan_amount, monthly_interest, term_months, repayment_type):
    schedule = []
    remaining_balance = loan_amount
    
    if repayment_type == "Fija":
        payment = loan_amount * (monthly_interest * (1 + monthly_interest)**term_months) / ((1 + monthly_interest)**term_months - 1)
        for month in range(1, term_months + 1):
            interest_payment = remaining_balance * monthly_interest
            principal_payment = payment - interest_payment
            remaining_balance -= principal_payment
            
            if month == term_months:  # Ajuste para el último mes (redondeo)
                principal_payment += remaining_balance
                remaining_balance = 0
                
            schedule.append({
                'month': month,
                'payment': payment,
                'principal': principal_payment,
                'interest': interest_payment,
                'remaining_balance': remaining_balance
            })
    else:  # Cuota decreciente
        principal_payment = loan_amount / term_months
        for month in range(1, term_months + 1):
            interest_payment = remaining_balance * monthly_interest
            payment = principal_payment + interest_payment
            remaining_balance -= principal_payment
            
            schedule.append({
                'month': month,
                'payment': payment,
                'principal': principal_payment,
                'interest': interest_payment,
                'remaining_balance': remaining_balance
            })
    
    return pd.DataFrame(schedule)

# Cálculo de estimación de ventas mensuales requeridas
def calculate_required_sales(amortization_df, retention_schedule, platform_share):
    # 1) Convertimos retention_schedule a lista de floats
    try:
        schedule = list(retention_schedule)
    except TypeError:
        # Si era un único float, repetimos ese valor para cada mes
        schedule = [float(retention_schedule)] * len(amortization_df)
    # 2) Si la lista es más corta que el número de meses, la extendemos con el último valor
    if len(schedule) < len(amortization_df):
        schedule.extend([schedule[-1]] * (len(amortization_df) - len(schedule)))

    monthly_requirements = []
    # 3) Iteramos y usamos índices Python int
    for _, row in amortization_df.iterrows():
        month = int(row['month'])          # convertimos a Python int
        payment = float(row['payment'])    # por si acaso viene como numpy.float64
        idx = min(month - 1, len(schedule) - 1)  # índice garantizado válido y entero
        retention_rate = schedule[idx]

        required_month_sales = payment / (retention_rate * platform_share)
        monthly_requirements.append({
            'month': month,
            'payment': payment,
            'retention_rate': retention_rate,
            'required_sales': required_month_sales
        })

    return pd.DataFrame(monthly_requirements)

def calculate_adjusted_pd(df, base_pd, risk_mult_low, risk_mult_medium, risk_mult_high):
    """Calcula PD ajustada por riesgo MCC"""
    def get_multiplier(calificacion):
        if pd.isna(calificacion):
            return risk_mult_medium  # Default medio
        elif calificacion <= 3:
            return risk_mult_low
        elif calificacion <= 6:
            return risk_mult_medium
        else:
            return risk_mult_high
    
    df['risk_multiplier'] = df['Calificación'].apply(get_multiplier)
    df['adjusted_pd'] = base_pd * df['risk_multiplier']
    
    # Asegurar que no exceda 100%
    df['adjusted_pd'] = df['adjusted_pd'].clip(upper=1.0)
    
    return df

# Filtrar negocios que califican
def filter_qualifying_businesses(df, min_months_active, min_weeks_active, monthly_threshold, min_antiguedad):
    return df[
        (df['meses_activos'] >= min_months_active) & 
        (df['semanas_activas'] >= min_weeks_active) & 
        (df['avg_monthly_sales'] >= monthly_threshold) &
        (df['antiguedad_meses'] >= min_antiguedad)
    ]

df = calculate_adjusted_pd(df, PD, risk_multiplier_low, risk_multiplier_medium, risk_multiplier_high)
# Ejecutar filtrado
df_qualify = filter_qualifying_businesses(df, min_months_active, min_weeks_active, monthly_threshold, min_antiguedad)

# KPIs principales
total_principal = len(df_qualify) * loan_amount
total_interest = monthly_payment * term_months * len(df_qualify) - total_principal
expected_loss = (df_qualify['adjusted_pd'] * LGD * loan_amount).sum()
total_operational_costs = operational_cost * len(df_qualify)
cost_of_funds_total = (cost_of_funds / 12) * total_principal * term_months

# ROI y tasa de equilibrio
net_income = total_interest - expected_loss - total_operational_costs - cost_of_funds_total
roi = net_income / total_principal if total_principal > 0 else 0
break_even_rate = (((cost_of_funds / 12) * term_months) + (operational_cost / loan_amount) + (PD * LGD)) / (1 - PD)

# Crear dos pestañas para separar los análisis
tab1, tab2, tab3 = st.tabs(["📊 Análisis de Cartera", "💰 Rentabilidad y Amortización", "🧮 Negocios Calificados"])

with tab1:
    # Primera fila: KPIs de cartera
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Negocios que Califican</div>
            <div class="metric-value">{len(df_qualify):,}</div>
            <div class="metric-change neutral-change">De {len(df):,} negocios ({len(df_qualify)/len(df)*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Total Principal</div>
            <div class="metric-value">Q {total_principal:,.0f}</div>
            <div class="metric-change neutral-change">Promedio Q {loan_amount:,.0f} por negocio</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Ingresos por Interés</div>
            <div class="metric-value">Q {total_interest:,.0f}</div>
            <div class="metric-change positive-change">Tasa efectiva: {(total_interest/total_principal)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Pérdida Esperada</div>
            <div class="metric-value">Q {expected_loss:,.0f}</div>
            <div class="metric-change negative-change">{PD*100:.1f}% probabilidad × {LGD*100:.0f}% severidad</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Segunda fila: Más KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">ROI Esperado</div>
            <div class="metric-value">{roi*100:.2f}%</div>
            <div class="metric-change {'positive-change' if roi >= min_roi else 'negative-change'}">
                {'✓ Supera' if roi >= min_roi else '✗ No alcanza'} ROI mínimo ({min_roi*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Tasa de Break-Even</div>
            <div class="metric-value">{break_even_rate*100:.2f}%</div>
            <div class="metric-change {'positive-change' if interest_rate/12*term_months > break_even_rate else 'negative-change'}">
                {'✓ Rentable' if interest_rate/12*term_months > break_even_rate else '✗ No rentable'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Costo de Fondeo</div>
            <div class="metric-value">Q {cost_of_funds_total:,.0f}</div>
            <div class="metric-change neutral-change">{cost_of_funds*100:.1f}% anual</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Costos Operativos</div>
            <div class="metric-value">Q {total_operational_costs:,.0f}</div>
            <div class="metric-change neutral-change">Q {operational_cost:.0f} por préstamo</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="subheader">Distribución por Segmentos</div>', unsafe_allow_html=True)
    
    # Análisis por segmentos (dos columnas)
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de calificaciones por actividad
        top_activities = df_qualify['Actividad'].value_counts().head(10)
        
        fig = px.bar(
            top_activities, 
            labels={'index': 'Actividad', 'value': 'Número de Negocios'},
            title="Top 10 Actividades de Negocios Calificados"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de distribución geográfica
        geo_dist = df_qualify['municipio'].value_counts().head(10)
        
        fig = px.bar(
            geo_dist, 
            labels={'index': 'municipio', 'value': 'Número de Negocios'},
            title="Top 10 Municipios de Negocios Calificados"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de ventas mensuales
    st.markdown('<div class="subheader">Distribución de Ventas Mensuales</div>', unsafe_allow_html=True)
    
    fig = px.histogram(
        df_qualify, 
        x='avg_monthly_sales',
        nbins=50,
        labels={'avg_monthly_sales': 'Ventas Mensuales Promedio (Q)'},
        title="Distribución de Ventas Mensuales de Negocios Calificados"
    )
    
    # Agregar línea vertical para el umbral
    fig.add_vline(
        x=monthly_threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Umbral: Q{monthly_threshold:,.0f}",
        annotation_position="top"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Después de los gráficos de actividad y municipio
    st.markdown('<div class="subheader">Análisis de Riesgo por MCC</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Distribución de niveles de riesgo
        risk_dist = df_qualify['Nivel de riesgo'].value_counts()
        
        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="Distribución por Nivel de Riesgo MCC"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # PD promedio por nivel de riesgo
        pd_by_risk = df_qualify.groupby('Nivel de riesgo')['adjusted_pd'].mean().reset_index()
        
        fig = px.bar(
            pd_by_risk,
            x='Nivel de riesgo',
            y='adjusted_pd',
            title="PD Promedio por Nivel de Riesgo",
            labels={'adjusted_pd': 'PD Ajustada (%)'}
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

    # Tabla de riesgo detallada
    st.markdown('<div class="subheader">Resumen de Riesgo por MCC</div>', unsafe_allow_html=True)

    risk_summary = df_qualify.groupby(['codigoMcc', 'Nivel de riesgo', 'Calificación']).agg({
        'nombre_negocio': 'count',
        'adjusted_pd': 'mean',
        'avg_monthly_sales': 'mean'
    }).rename(columns={
        'nombre_negocio': 'Cantidad Negocios',
        'adjusted_pd': 'PD Promedio',
        'avg_monthly_sales': 'Ventas Promedio'
    }).reset_index()

    st.dataframe(
        risk_summary.style.format({
            'PD Promedio': '{:.2%}',
            'Ventas Promedio': 'Q{:,.0f}'
        }),
        use_container_width=True
    )

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Cuota Mensual</div>
            <div class="metric-value">Q {monthly_payment:.2f}</div>
            <div class="metric-change neutral-change">Para préstamo de Q {loan_amount:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_payment = monthly_payment * term_months
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Total a Pagar</div>
            <div class="metric-value">Q {total_payment:.2f}</div>
            <div class="metric-change neutral-change">Interés total: Q {total_payment - loan_amount:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        effective_rate = ((total_payment / loan_amount) - 1) * 100
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Tasa Efectiva Total</div>
            <div class="metric-value">{effective_rate:.2f}%</div>
            <div class="metric-change neutral-change">Para {term_months} meses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cost_benefit = total_interest - expected_loss - cost_of_funds_total - total_operational_costs
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Ganancia Neta</div>
            <div class="metric-value">Q {net_income:,.0f}</div>
            <div class="metric-change {'positive-change' if net_income > 0 else 'negative-change'}">
                {net_income/total_principal*100:.2f}% sobre capital
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Preparación de datos para amortización
    amortization_df = generate_amortization_schedule(loan_amount, monthly_interest, term_months, repayment_type)
    required_sales_df = calculate_required_sales(amortization_df, retention_schedule, platform_share)
    
    # Análisis de rentabilidad
    st.markdown('<div class="subheader">Análisis de Rentabilidad del Préstamo</div>', unsafe_allow_html=True)
    
    # Rentabilidad por préstamo individual
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de amortización
        amort_fig = go.Figure()
        
        amort_fig.add_trace(go.Bar(
            x=amortization_df['month'],
            y=amortization_df['principal'],
            name='Principal',
            marker_color='#3B82F6'
        ))
        
        amort_fig.add_trace(go.Bar(
            x=amortization_df['month'],
            y=amortization_df['interest'],
            name='Interés',
            marker_color='#10B981'
        ))
        
        amort_fig.update_layout(
            title='Desglose de Pagos Mensuales',
            xaxis_title='Mes',
            yaxis_title='Monto (Q)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(amort_fig, use_container_width=True)
    
    with col2:
        # Gráfico de saldo restante
        balance_fig = px.line(
            amortization_df,
            x='month',
            y='remaining_balance',
            labels={'month': 'Mes', 'remaining_balance': 'Saldo Restante (Q)'},
            title='Evolución del Saldo Restante',
            height=400
        )
        
        balance_fig.update_traces(line_color='#3B82F6')
        
        st.plotly_chart(balance_fig, use_container_width=True)
    
    # Requerimientos de ventas y retención
    st.markdown('<div class="subheader">Requisitos de Ventas Mensuales</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de ventas requeridas por mes
        sales_fig = px.line(
            required_sales_df,
            x='month',
            y='required_sales',
            labels={'month': 'Mes', 'required_sales': 'Ventas Requeridas (Q)'},
            title='Ventas Mensuales Requeridas',
            height=400
        )
        
        sales_fig.update_traces(line_color='#10B981')
        
        # Línea horizontal para promedio
        sales_fig.add_hline(
            y=required_sales_df['required_sales'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Promedio: Q{required_sales_df['required_sales'].mean():,.0f}",
            annotation_position="top right"
        )
        
        st.plotly_chart(sales_fig, use_container_width=True)
    
    with col2:
        # Gráfico de tasas de retención
        retention_fig = px.line(
            required_sales_df,
            x='month',
            y='retention_rate',
            labels={'month': 'Mes', 'retention_rate': 'Tasa de Retención (%)'},
            title='Tasas de Retención Mensuales',
            height=400
        )
        
        retention_fig.update_traces(line_color='#6366F1')
        retention_fig.update_layout(yaxis_tickformat='.1%')
        
        st.plotly_chart(retention_fig, use_container_width=True)
    
    # Tabla de amortización
    st.markdown('<div class="subheader">Tabla de Amortización Detallada</div>', unsafe_allow_html=True)
    
    # Combinar datos de amortización y ventas requeridas
    full_schedule = pd.merge(
        amortization_df,
        required_sales_df[['month', 'retention_rate', 'required_sales']],
        on='month'
    )
    
    # Formatear para presentación
    display_cols = {
        'month': 'Mes',
        'payment': 'Cuota Total (Q)',
        'principal': 'Principal (Q)',
        'interest': 'Interés (Q)',
        'remaining_balance': 'Saldo Restante (Q)',
        'retention_rate': 'Tasa Retención (%)',
        'required_sales': 'Ventas Requeridas (Q)'
    }
    
    display_df = full_schedule[display_cols.keys()].copy()
    display_df.columns = display_cols.values()
    
    # Formatear columnas numéricas
    for col in display_df.columns:
        if col == 'Mes':
            continue
        elif col == 'Tasa Retención (%)':
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"Q {x:,.2f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Análisis de Umbral
    st.markdown('<div class="subheader">Análisis de Sensibilidad</div>', unsafe_allow_html=True)
    
    # Función para calcular umbral de ventas con diferentes parámetros
    def calculate_threshold(loan_amount, term, interest, retention, platform_share):
        monthly_int = interest / 12
        if repayment_type == "Fija":
            payment = loan_amount * (monthly_int * (1 + monthly_int)**term) / ((1 + monthly_int)**term - 1)
        else:
            principal = loan_amount / term
            interest_first = loan_amount * monthly_int
            payment = (principal + interest_first)
        
        total_payment = payment * term
        required_sales = total_payment / (retention * platform_share)
        monthly_sales = required_sales / term
        return monthly_sales
    
    # Crear matriz de sensibilidad: Retención vs. Plazo
    retention_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    term_values = [3, 6, 9, 12]
    
    sensitivity_data = []
    
    for ret in retention_values:
        row = {}
        row['Retención'] = f"{ret*100:.0f}%"
        
        for term in term_values:
            threshold = calculate_threshold(loan_amount, term, interest_rate, ret, platform_share)
            row[f"{term} meses"] = threshold
        
        sensitivity_data.append(row)
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    # Mostrar tabla de sensibilidad
    st.dataframe(sensitivity_df.style.format({col: "Q{:,.0f}" for col in sensitivity_df.columns if col != "Retención"}), use_container_width=True)
    
    # Gráfico de sensibilidad
    sensitivity_plot_data = []
    for term in term_values:
        for ret in retention_values:
            threshold = calculate_threshold(loan_amount, term, interest_rate, ret, platform_share)
            sensitivity_plot_data.append({
                'Retención': f"{ret*100:.0f}%",
                'Plazo (meses)': term,
                'Umbral Ventas Mensuales': threshold
            })
    
    sens_df = pd.DataFrame(sensitivity_plot_data)
    
    sens_fig = px.line(
        sens_df,
        x='Retención',
        y='Umbral Ventas Mensuales',
        color='Plazo (meses)',
        title='Sensibilidad del Umbral de Ventas',
        labels={'Umbral Ventas Mensuales': 'Ventas Mensuales Requeridas (Q)'}
    )
    
    st.plotly_chart(sens_fig, use_container_width=True)

with tab3:
    # Título de la pestaña
    st.markdown(
        '<div class="subheader">Negocios que Cumplen los Criterios</div>',
        unsafe_allow_html=True
    )

    # Si no hay ninguno, aviso; si hay, mostramos el dataframe completo (o sólo las columnas clave)
    if df_qualify.empty:
        st.info("No hay negocios que cumplan los filtros actuales. Ajusta los parámetros para ver resultados.")
    else:
        # Opcional: selecciona sólo unas columnas para evitar mostrar 30+ columnas de golpe
        cols = [
            'nombre_negocio',
            'avg_monthly_sales',
            'meses_activos',
            'semanas_activas',
            'antiguedad_meses'
        ]
        st.dataframe(
            df_qualify[cols].reset_index(drop=True),
            use_container_width=True,
            height=600   # ajusta la altura si lo ves muy comprimido
        )
    
    # En tab3, actualiza las columnas
    cols = [
        'nombre_negocio',
        'avg_monthly_sales',
        'meses_activos',
        'semanas_activas',
        'antiguedad_meses',
        'codigoMcc',
        'Nivel de riesgo',
        'Calificación',
        'adjusted_pd'
    ]

    # Formatear la tabla
    display_df = df_qualify[cols].copy()
    display_df['adjusted_pd'] = display_df['adjusted_pd'].apply(lambda x: f"{x:.2%}")
    display_df.rename(columns={
        'adjusted_pd': 'PD Ajustada (%)',
        'Calificación': 'Calif. Riesgo'
    }, inplace=True)

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        height=600
    )
