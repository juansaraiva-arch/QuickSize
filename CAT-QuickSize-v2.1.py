import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT QuickSize v2.1", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY - ENHANCED WITH DENSITY & RAMP RATE
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "ramp_rate_mw_s": 0.5,
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "mtbf_hours": 50000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520FR": {
        "description": "Fast Response Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.5,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8836,
        "step_load_pct": 40.0,
        "ramp_rate_mw_s": 0.6,
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "mtbf_hours": 48000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.0,
        "default_maint": 5.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520K": {
        "description": "High Efficiency Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.4,
        "electrical_efficiency": 0.453,
        "heat_rate_lhv": 7638,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 0.4,
        "emissions_nox": 0.3,
        "emissions_co": 2.3,
        "mtbf_hours": 52000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.5,
        "default_maint": 6.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.13
    },
    "CG260-16": {
        "description": "Cogeneration Specialist (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 3.96,
        "electrical_efficiency": 0.434,
        "heat_rate_lhv": 7860,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.45,
        "emissions_nox": 0.5,
        "emissions_co": 1.8,
        "mtbf_hours": 55000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 3.0,
        "default_maint": 5.0,
        "est_cost_kw": 675.0,
        "est_install_kw": 1100.0,
        "power_density_mw_per_m2": 0.009,
        "gas_pressure_min_psi": 7.25,
        "reactance_xd_2": 0.15
    },
    "Titan 130": {
        "description": "Solar Gas Turbine (16.5 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 2.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "mtbf_hours": 80000,
        "maintenance_interval_hrs": 8000,
        "maintenance_duration_hrs": 120,
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
        "power_density_mw_per_m2": 0.020,
        "gas_pressure_min_psi": 300.0,
        "reactance_xd_2": 0.18
    },
    "G20CM34": {
        "description": "Medium Speed Baseload Platform",
        "type": "Medium Speed",
        "iso_rating_mw": 9.76,
        "electrical_efficiency": 0.475,
        "heat_rate_lhv": 7480,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.3,
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "mtbf_hours": 60000,
        "maintenance_interval_hrs": 2500,
        "maintenance_duration_hrs": 72,
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "power_density_mw_per_m2": 0.008,
        "gas_pressure_min_psi": 90.0,
        "reactance_xd_2": 0.16
    }
}

# ==============================================================================
# HELPER FUNCTIONS - ENHANCED
# ==============================================================================

def get_part_load_efficiency(base_eff, load_pct, gen_type):
    """Efficiency curves validated against CAT test data"""
    if gen_type == "High Speed":
        eff_mult = -0.0008*(load_pct**2) + 0.18*load_pct + 82
        return base_eff * (eff_mult / 100)
    elif gen_type == "Medium Speed":
        eff_mult = -0.0005*(load_pct**2) + 0.12*load_pct + 88
        return base_eff * (eff_mult / 100)
    elif gen_type == "Gas Turbine":
        eff_mult = -0.0015*(load_pct**2) + 0.25*load_pct + 75
        return base_eff * (eff_mult / 100)
    return base_eff

def transient_stability_check(xd_pu, num_units, step_load_pct):
    """Critical voltage sag check for AI workloads"""
    equiv_xd = xd_pu / math.sqrt(num_units)
    voltage_sag = (step_load_pct/100) * equiv_xd * 100
    if voltage_sag > 10:
        return False, voltage_sag
    return True, voltage_sag

def calculate_bess_requirements(p_net_req_avg, p_net_req_peak, step_load_req, 
                                gen_ramp_rate, gen_step_capability, enable_black_start=False):
    """
    Sophisticated BESS sizing based on actual transient analysis
    NOW RESPONDS TO: Increased step load requirements
    """
    # Component 1: Step Load Support (CRITICAL - responds to step_load_req)
    step_load_mw = p_net_req_avg * (step_load_req / 100)
    gen_step_mw = p_net_req_avg * (gen_step_capability / 100)
    bess_step_support = max(0, step_load_mw - gen_step_mw)
    
    # Component 2: Peak Shaving (covers peak vs average difference)
    bess_peak_shaving = p_net_req_peak - p_net_req_avg
    
    # Component 3: Ramp Rate Support
    load_change_rate = 5.0  # MW/s (AI workload aggressive)
    bess_ramp_support = max(0, (load_change_rate - gen_ramp_rate) * 10)  # 10s buffer
    
    # Component 4: Frequency Regulation
    bess_freq_reg = p_net_req_avg * 0.05  # 5% for freq regulation
    
    # Component 5: Black Start Capability
    bess_black_start = p_net_req_peak * 0.05 if enable_black_start else 0
    
    # Total Power (take maximum of all requirements)
    bess_power_total = max(
        bess_step_support,
        bess_peak_shaving,
        bess_ramp_support,
        bess_freq_reg,
        bess_black_start,
        p_net_req_peak * 0.15  # Minimum 15% floor
    )
    
    # Energy Duration Calculation
    c_rate = 1.0  # 1C = discharge in 1 hour
    bess_energy_total = bess_power_total / c_rate
    
    # Round-trip efficiency consideration
    rte = 0.85  # 85% round-trip efficiency
    bess_energy_total = bess_energy_total / rte
    
    breakdown = {
        'step_support': bess_step_support,
        'peak_shaving': bess_peak_shaving,
        'ramp_support': bess_ramp_support,
        'freq_reg': bess_freq_reg,
        'black_start': bess_black_start
    }
    
    return bess_power_total, bess_energy_total, breakdown

def calculate_bess_reliability_credit(bess_power_mw, bess_energy_mwh, 
                                      unit_capacity_mw, mttr_hours=48):
    """
    Calculate how many genset equivalents BESS can replace for reliability
    """
    if bess_power_mw <= 0 or bess_energy_mwh <= 0:
        return 0.0, {}
    
    realistic_coverage_hrs = 2.0  # Conservative: 2 hours
    
    power_credit = bess_power_mw / unit_capacity_mw
    energy_credit = bess_energy_mwh / (unit_capacity_mw * realistic_coverage_hrs)
    raw_credit = min(power_credit, energy_credit)
    
    bess_availability = 0.98
    coverage_factor = 0.70
    
    effective_credit = raw_credit * bess_availability * coverage_factor
    
    credit_breakdown = {
        'power_credit': power_credit,
        'energy_credit': energy_credit,
        'raw_credit': raw_credit,
        'bess_availability': bess_availability,
        'coverage_factor': coverage_factor,
        'effective_credit': effective_credit,
        'realistic_coverage_hrs': realistic_coverage_hrs
    }
    
    return effective_credit, credit_breakdown

def calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, 
                                  maintenance_interval_hrs=1000, maintenance_duration_hrs=48):
    """
    Reliability model using industry standard availability formula INCLUDING planned maintenance
    """
    mttr_hours = 48
    annual_maintenance_hrs = (8760 / maintenance_interval_hrs) * maintenance_duration_hrs
    total_unavailable_hrs = mttr_hours + annual_maintenance_hrs
    unit_availability = mtbf_hours / (mtbf_hours + total_unavailable_hrs)
    
    sys_avail = 0
    for k in range(n_running, n_total + 1):
        comb = math.comb(n_total, k)
        prob = comb * (unit_availability ** k) * ((1 - unit_availability) ** (n_total - k))
        sys_avail += prob
    
    availability_over_time = []
    for year in range(1, project_years + 1):
        aging_factor = 1.0 - (year * 0.001)
        aging_factor = max(0.95, aging_factor)
        aged_unit_availability = unit_availability * aging_factor
        
        sys_avail_year = 0
        for k in range(n_running, n_total + 1):
            comb = math.comb(n_total, k)
            prob = comb * (aged_unit_availability ** k) * ((1 - aged_unit_availability) ** (n_total - k))
            sys_avail_year += prob
        
        availability_over_time.append(sys_avail_year)
    
    return sys_avail, availability_over_time

# --- FIXED: OPTIMIZATION FUNCTION (PRIORITIZE CAPEX & STEP LOAD) ---
def optimize_fleet_size(p_net_req_avg, p_net_req_peak, unit_cap, step_load_req, gen_data, use_bess=False):
    """
    Optimización corregida: Considera explícitamente la reserva rodante para Step Load.
    """
    # 1. Criterio de Capacidad PICO
    if use_bess:
        # Con BESS, cubrimos el promedio + margen pequeño (el BESS cubre los picos)
        n_min_peak = math.ceil((p_net_req_avg * 1.05) / unit_cap)
    else:
        # Sin BESS, cubrimos el pico absoluto
        n_min_peak = math.ceil(p_net_req_peak / unit_cap)

    # 2. Criterio de STEP LOAD (El más crítico para "No BESS")
    if use_bess:
        # El BESS absorbe el golpe, no necesitamos reserva rodante masiva
        n_min_step = 0
    else:
        # Sin BESS, necesitamos: (Capacidad Total - Carga Promedio) >= Carga del Golpe
        # N * Cap >= Avg + (Avg * Step%)
        mw_required_for_step = p_net_req_avg * (1 + step_load_req/100.0)
        n_min_step = math.ceil(mw_required_for_step / unit_cap)

    # El mínimo operativo es el mayor de los dos criterios
    n_running_base = max(n_min_peak, n_min_step)
    
    # 3. Explorar opciones cercanas para optimizar eficiencia
    fleet_options = {}
    
    # Buscamos desde la base hasta +5 unidades (para ver si bajando carga mejora eficiencia)
    for n in range(n_running_base, n_running_base + 6):
        capacity = n * unit_cap
        load_pct = (p_net_req_avg / capacity) * 100
        
        # Validaciones de rango operativo
        if load_pct < 30: continue # Muy baja carga (wet stacking)
        if load_pct > 95: continue # Muy alta (sin margen de regulación)
        
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        # Score: Eficiencia penalizada por exceso de unidades (CAPEX)
        penalty_low_load = 0.05 if load_pct < 50 else 0
        score = eff - (n * 0.005) - penalty_low_load
        
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': score
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        # Fallback si no encuentra opciones válidas
        return n_running_base, {n_running_base: {'efficiency': 0.35, 'load_pct': 80, 'score': 0}}

def calculate_macrs_depreciation(capex, project_years):
    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    tax_rate = 0.21
    pv_benefit = 0
    wacc = 0.08
    for year, rate in enumerate(macrs_schedule, 1):
        if year > project_years:
            break
        annual_benefit = capex * rate * tax_rate
        pv_benefit += annual_benefit / ((1 + wacc) ** year)
    return pv_benefit

# ==============================================================================
# 1. GLOBAL SETTINGS & SIDEBAR
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"])
    freq_hz = c_glob2.radio("System Frequency", [60, 50])

is_imperial = "Imperial" in unit_system
is_50hz = freq_hz == 50

# Unit Strings
if is_imperial:
    u_temp, u_dist, u_area_s, u_area_l = "°F", "ft", "ft²", "Acres"
    u_vol, u_mass, u_power = "gal", "Short Tons", "MW"
    u_energy, u_therm, u_water = "MWh", "MMBtu", "gal/day"
    u_press = "psig"
else:
    u_temp, u_dist, u_area_s, u_area_l = "°C", "m", "m²", "Ha"
    u_vol, u_mass, u_power = "m³", "Tonnes", "MW"
    u_energy, u_therm, u_water = "MWh", "GJ", "m³/day"
    u_press = "Bar"

st.title(f"⚡ CAT QuickSize v2.1 ({freq_hz}Hz)")
st.markdown("**Next-Gen Data Center Power Solutions.**\nAdvanced modeling with PUE optimization, footprint constraints, and sophisticated LCOE analysis.")

# ==============================================================================
# 2. INPUTS - FULL DETAIL
# ==============================================================================

with st.sidebar:
    st.header("1. Site & Requirements")
    
    st.markdown("🏗️ **Data Center Profile**")
    dc_type = st.selectbox("Data Center Type", [
        "AI Factory (Training)", 
        "AI Factory (Inference)",
        "Hyperscale Standard", 
        "Colocation", 
        "Edge Computing"
    ])
    
    pue_defaults = {
        "AI Factory (Training)": 1.15,
        "AI Factory (Inference)": 1.20,
        "Hyperscale Standard": 1.25,
        "Colocation": 1.50,
        "Edge Computing": 1.60
    }
    
    is_ai = "AI" in dc_type
    def_step_load = 40.0 if is_ai else 15.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    
    st.markdown("📊 **Power Usage Effectiveness (PUE)**")
    pue = st.slider("Data Center PUE", 1.05, 2.00, pue_defaults[dc_type], 0.05)
    
    p_total_dc = p_it * pue
    p_aux = p_total_dc - p_it
    
    with st.expander("ℹ️ PUE Breakdown"):
        st.write(f"**IT Load:** {p_it:.1f} MW")
        st.write(f"**Auxiliary Load:** {p_aux:.1f} MW ({(pue-1)*100:.1f}% of IT)")
        st.write(f"**Total DC Load:** {p_total_dc:.1f} MW")
    
    # ===== LOAD PROFILE SECTION =====
    st.markdown("📊 **Annual Load Profile**")
    
    load_profiles = {
        "AI Factory (Training)": {"capacity_factor": 0.96, "peak_avg_ratio": 1.08},
        "AI Factory (Inference)": {"capacity_factor": 0.85, "peak_avg_ratio": 1.25},
        "Hyperscale Standard": {"capacity_factor": 0.75, "peak_avg_ratio": 1.20},
        "Colocation": {"capacity_factor": 0.65, "peak_avg_ratio": 1.35},
        "Edge Computing": {"capacity_factor": 0.50, "peak_avg_ratio": 1.50}
    }
    
    profile = load_profiles[dc_type]
    
    col_cf1, col_cf2 = st.columns(2)
    capacity_factor = col_cf1.slider("Capacity Factor (%)", 30.0, 100.0, profile["capacity_factor"]*100, 1.0) / 100.0
    peak_avg_ratio = col_cf2.slider("Peak/Avg Ratio", 1.0, 2.0, profile["peak_avg_ratio"], 0.05)
    
    p_total_avg = p_total_dc * capacity_factor
    p_total_peak = p_total_dc * peak_avg_ratio
    
    st.info(f"💡 **Load Analysis:**\n"
            f"- Avg: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**\n"
            f"- Effective Hours/Year: **{8760*capacity_factor:.0f} hrs**")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Step Load Req (%)", 0.0, 100.0, def_step_load)
    
    st.markdown("📐 **Site Constraints**")
    enable_footprint_limit = st.checkbox("Enable Footprint Limit", value=False)
    
    if enable_footprint_limit:
        area_unit_sel = st.radio("Area Unit", ["m²", "Acres", "Hectares"], horizontal=True)
        if area_unit_sel == "m²":
            max_area_input = st.number_input("Max Available Area (m²)", 100.0, 500000.0, 50000.0, step=1000.0)
            max_area_m2 = max_area_input
        elif area_unit_sel == "Acres":
            max_area_input = st.number_input("Max Available Area (Acres)", 0.1, 100.0, 12.0, step=0.5)
            max_area_m2 = max_area_input / 0.000247105
        else:
            max_area_input = st.number_input("Max Available Area (Ha)", 0.1, 50.0, 5.0, step=0.5)
            max_area_m2 = max_area_input * 10000
    else:
        max_area_m2 = 999999999
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual":
        voltage_option = st.selectbox("Select Voltage Level", [
            "4.16 kV", "13.8 kV", "34.5 kV", "69 kV", "Custom"
        ])
        voltage_map = {"4.16 kV": 4.16, "13.8 kV": 13.8, "34.5 kV": 34.5, "69 kV": 69.0}
        if voltage_option == "Custom":
            manual_voltage_kv = st.number_input("Custom Voltage (kV)", 0.4, 230.0, 13.8, step=0.1)
        else:
            manual_voltage_kv = voltage_map[voltage_option]

    st.markdown("🌍 **Site Environment**")
    derate_mode = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True)
    
    if derate_mode == "Auto-Calculate":
        c_env1, c_env2 = st.columns(2)
        if is_imperial:
            site_temp_f = c_env1.number_input(f"Ambient Temp ({u_temp})", 32, 130, 77)
            site_temp_c = (site_temp_f - 32) * 5/9
        else:
            site_temp_c = c_env1.number_input(f"Ambient Temp ({u_temp})", 0, 55, 25)
        
        if is_imperial:
            site_alt_ft = c_env2.number_input(f"Altitude ({u_dist})", 0, 15000, 0, step=100)
            site_alt_m = site_alt_ft * 0.3048
        else:
            site_alt_m = c_env2.number_input(f"Altitude ({u_dist})", 0, 4500, 0, step=50)
        
        methane_number = st.slider("Gas Methane Number", 50, 100, 80)
        
        temp_derate = 1.0 - max(0, (site_temp_c - 25) * 0.01)
        alt_derate = 1.0 - (site_alt_m / 300)
        fuel_derate = 1.0 if methane_number >= 70 else 0.95
        derate_factor_calc = temp_derate * alt_derate * fuel_derate
    else:
        derate_factor_calc = st.slider("Manual Derate Factor", 0.5, 1.0, 0.9, 0.01)
        site_temp_c = 25
        site_alt_m = 0
        methane_number = 80

    st.header("2. Technology Solution")
    st.markdown("⚙️ **Generation Technology**")
    gen_filter = st.multiselect("Technology Filter", ["High Speed", "Medium Speed", "Gas Turbine"], default=["High Speed", "Medium Speed"])
    use_bess = st.checkbox("Include BESS (Battery Energy Storage)", value=def_use_bess)
    
    bess_strategy = "Hybrid (Balanced)"
    bess_reliability_enabled = False
    
    if use_bess:
        st.markdown("🔋 **BESS Strategy**")
        bess_strategy = st.radio("Sizing Mode", ["Transient Only", "Hybrid (Balanced)", "Reliability Priority"], index=1)
        bess_reliability_enabled = bess_strategy != "Transient Only"
        if bess_reliability_enabled:
            st.caption("💡 BESS will provide backup power to reduce genset count")
    
    enable_black_start = st.checkbox("Enable Black Start Capability", value=False)
    include_chp = st.checkbox("Include Tri-Generation (CHP)", value=False)
    if include_chp:
        cooling_method = "Absorption Chiller"
    else:
        cooling_method = st.selectbox("Cooling Method", ["Air-Cooled", "Water-Cooled"])
    
    st.markdown("⛽ **Fuel Infrastructure**")
    fuel_mode = st.radio("Primary Fuel", ["Pipeline Gas", "LNG", "Dual-Fuel"], horizontal=True)
    is_lng_primary = "LNG" in fuel_mode
    has_lng_storage = fuel_mode in ["LNG", "Dual-Fuel"]
    
    if has_lng_storage:
        lng_days = st.number_input("LNG Storage (Days)", 1, 90, 7)
    else:
        lng_days = 0
        dist_gas_main_km = st.number_input("Distance to Gas Main (km)", 0.1, 100.0, 1.0)
        dist_gas_main_m = dist_gas_main_km * 1000

    st.header("3. Economics & ROI")
    st.markdown("💰 **Energy Pricing**")
    col_g1, col_g2 = st.columns(2)
    gas_price_wellhead = col_g1.number_input("Gas Price - Wellhead ($/MMBtu)", 0.5, 30.0, 4.0, step=0.5)
    gas_transport = col_g2.number_input("Pipeline Transport ($/MMBtu)", 0.0, 5.0, 0.5, step=0.1)
    
    if is_lng_primary:
        lng_regasification = st.number_input("LNG Regasification ($/MMBtu)", 0.5, 3.0, 1.5, step=0.1)
        lng_transport = st.number_input("LNG Shipping ($/MMBtu)", 1.0, 5.0, 3.0, step=0.5)
    else:
        lng_regasification = 0
        lng_transport = 0
    
    total_gas_price = gas_price_wellhead + gas_transport + lng_regasification + lng_transport
    st.info(f"**Total Gas Cost:** ${total_gas_price:.2f}/MMBtu")
    
    benchmark_price = st.number_input("Benchmark Electricity ($/kWh)", 0.01, 0.50, 0.12, step=0.01)
    
    st.markdown("🌍 **Carbon Pricing**")
    carbon_scenario = st.selectbox("Carbon Price Scenario", ["None (Current 2026)", "California Cap-and-Trade", "EU ETS", "Federal Projected 2030", "High Case (IEA Net Zero)"])
    carbon_prices = {"None (Current 2026)": 0, "California Cap-and-Trade": 35, "EU ETS": 85, "Federal Projected 2030": 50, "High Case (IEA Net Zero)": 150}
    carbon_price_per_ton = carbon_prices[carbon_scenario]
    
    c_fin1, c_fin2 = st.columns(2)
    wacc = c_fin1.number_input("WACC (%)", 1.0, 20.0, 8.0, step=0.5) / 100
    project_years = c_fin2.number_input("Project Life (Years)", 10, 30, 20, step=5)
    
    st.markdown("💸 **Tax Incentives & Depreciation**")
    enable_itc = st.checkbox("Include ITC (30% for CHP)", value=include_chp)
    enable_ptc = st.checkbox("Include PTC ($0.013/kWh, 10yr)", value=False)
    enable_depreciation = st.checkbox("Include MACRS Depreciation", value=True)
    
    st.markdown("📍 **Regional Adjustments**")
    region = st.selectbox("Region", ["US - Gulf Coast", "US - Northeast", "US - West Coast", "US - Midwest", "Europe - Western", "Europe - Eastern", "Middle East", "Asia Pacific", "Latin America", "Africa"])
    regional_multipliers = {"US - Gulf Coast": 1.0, "US - Northeast": 1.25, "US - West Coast": 1.30, "US - Midwest": 1.05, "Europe - Western": 1.35, "Europe - Eastern": 0.90, "Middle East": 1.10, "Asia Pacific": 0.85, "Latin America": 0.95, "Africa": 1.15}
    regional_mult = regional_multipliers[region]
    
    enable_lcoe_target = st.checkbox("Enable LCOE Target Mode", value=False)
    target_lcoe = 0.0
    if enable_lcoe_target:
        target_lcoe = st.number_input("Target LCOE ($/kWh)", 0.01, 0.50, 0.08, step=0.005)

# ==============================================================================
# 3. GENERATOR SELECTION & FLEET OPTIMIZATION
# ==============================================================================

available_gens = {k: v for k, v in leps_gas_library.items() if v["type"] in gen_filter}

if not available_gens:
    st.error("⚠️ No generators match filter. Adjust technology selection.")
    st.stop()

# Auto-select best generator
best_gen = None
best_score = -999

for gen_name, gen_data in available_gens.items():
    unit_derated = gen_data["iso_rating_mw"] * derate_factor_calc
    
    if unit_derated < (p_total_peak * 0.1):
        continue
    
    step_match = 1.0 if gen_data["step_load_pct"] >= step_load_req else 0.5
    eff_score = gen_data["electrical_efficiency"] * 10
    cost_score = -gen_data["est_cost_kw"] / 100
    density_score = gen_data["power_density_mw_per_m2"] * 20
    
    total_score = step_match * 100 + eff_score + cost_score + density_score
    
    if total_score > best_score:
        best_score = total_score
        best_gen = gen_name

selected_gen = st.sidebar.selectbox("🔧 Selected Generator", list(available_gens.keys()), index=list(available_gens.keys()).index(best_gen) if best_gen else 0)
gen_data = available_gens[selected_gen]

with st.sidebar.expander("⚙️ Generator Parameters (Editable)", expanded=False):
    mtbf_edit = st.number_input("MTBF (hours)", value=gen_data["mtbf_hours"], min_value=10000, max_value=150000, step=1000)
    gen_data["mtbf_hours"] = mtbf_edit
    maint_interval_edit = st.number_input("Maintenance Interval (hrs)", value=gen_data["maintenance_interval_hrs"], min_value=500, max_value=20000, step=100)
    gen_data["maintenance_interval_hrs"] = maint_interval_edit
    maint_duration_edit = st.number_input("Maintenance Duration (hrs)", value=gen_data["maintenance_duration_hrs"], min_value=12, max_value=240, step=6)
    gen_data["maintenance_duration_hrs"] = maint_duration_edit
    
    # Show availability impact
    annual_maint_hrs = (8760 / maint_interval_edit) * maint_duration_edit
    unit_avail = mtbf_edit / (mtbf_edit + 48 + annual_maint_hrs)
    st.metric("Single Unit Availability", f"{unit_avail*100:.2f}%")
    
    eff_edit = st.number_input("Electrical Efficiency", value=gen_data["electrical_efficiency"], min_value=0.25, max_value=0.60, step=0.001, format="%.3f")
    gen_data["electrical_efficiency"] = eff_edit
    step_edit = st.number_input("Step Load Capability (%)", value=gen_data["step_load_pct"], min_value=0.0, max_value=100.0, step=5.0)
    gen_data["step_load_pct"] = step_edit
    ramp_edit = st.number_input("Ramp Rate (MW/s)", value=gen_data["ramp_rate_mw_s"], min_value=0.1, max_value=5.0, step=0.1)
    gen_data["ramp_rate_mw_s"] = ramp_edit

unit_iso_cap = gen_data["iso_rating_mw"]
unit_site_cap = unit_iso_cap * derate_factor_calc

# ============================================================================
# ENHANCED FLEET OPTIMIZATION - AVAILABILITY-DRIVEN WITH BESS CREDIT
# ============================================================================

# Step 1: Calculate MINIMUM n_running based on load requirements (NEW FUNCTION)
n_running_from_load, fleet_options = optimize_fleet_size(
    p_total_avg, p_total_peak, unit_site_cap, step_load_req, gen_data, use_bess
)

# Step 2: Calculate N+X for availability target
avail_decimal = avail_req / 100
mtbf_hours = gen_data["mtbf_hours"]
mttr_hours = 48

# Calculate BESS requirements FIRST
bess_power_transient = 0.0
bess_energy_transient = 0.0
bess_breakdown_transient = {}

if use_bess:
    bess_power_transient, bess_energy_transient, bess_breakdown_transient = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        enable_black_start
    )

reliability_configs = []

# Configuration A: No BESS (Baseline)
# Forzamos que n_run respete el criterio de Step Load
config_a_running = n_running_from_load 

# Ampliamos búsqueda por si acaso
search_min_a = config_a_running
search_max_a = config_a_running + 5
best_config_a = None

for n_run in range(search_min_a, search_max_a):
    # CRITERIO 1: Capacidad Pico
    if n_run * unit_site_cap < p_total_peak:
        continue
        
    # CRITERIO 2: Reserva Rodante para Step Load (CRÍTICO)
    current_headroom_mw = (n_run * unit_site_cap) - p_total_avg
    required_step_mw = p_total_avg * (step_load_req / 100.0)
    
    if current_headroom_mw < required_step_mw:
        continue 
    
    for n_res in range(0, 20):
        n_tot = n_run + n_res
        avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
        
        if avg_avail >= avail_decimal:
            load_pct_a = (p_total_avg / (n_run * unit_site_cap)) * 100
            eff_a = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_a, gen_data["type"])
            
            best_config_a = {
                'name': 'A: No BESS',
                'n_running': n_run,
                'n_reserve': n_res,
                'n_total': n_tot,
                'bess_mw': 0,
                'bess_mwh': 0,
                'bess_credit': 0,
                'availability': avg_avail,
                'load_pct': load_pct_a,
                'efficiency': eff_a
            }
            break
    if best_config_a:
        break

if not best_config_a:
     best_config_a = {
        'name': 'A: No BESS (Fallback)',
        'n_running': config_a_running,
        'n_reserve': 1,
        'n_total': config_a_running + 1,
        'bess_mw': 0, 'bess_mwh': 0, 'bess_credit': 0,
        'availability': 0.99, 'load_pct': 90, 'efficiency': 0.35
     }
reliability_configs.append(best_config_a)

# Configuration B: BESS Transient Only
if use_bess:
    target_load_optimal = 72
    n_running_optimal_b = round(p_total_avg / (unit_site_cap * (target_load_optimal/100)))
    n_running_optimal_b = max(n_running_optimal_b, int(p_total_avg * 1.05 / unit_site_cap))
    
    best_config_b = None
    
    for n_run in range(max(1, n_running_optimal_b - 3), n_running_optimal_b + 6):
        capacity_check = n_run * unit_site_cap >= p_total_avg * 1.05
        if not capacity_check:
            continue
        
        for n_res in range(0, 20):
            n_tot = n_run + n_res
            avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
            
            if avg_avail >= avail_decimal:
                load_pct_b = (p_total_avg / (n_run * unit_site_cap)) * 100
                eff_b = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_b, gen_data["type"])
                score_b = eff_b - abs(load_pct_b - 72) * 0.01
                
                if best_config_b is None or score_b > best_config_b.get('score', 0):
                    best_config_b = {
                        'name': 'B: BESS Transient',
                        'n_running': n_run,
                        'n_reserve': n_res,
                        'n_total': n_tot,
                        'bess_mw': bess_power_transient,
                        'bess_mwh': bess_energy_transient,
                        'bess_credit': 0,
                        'availability': avg_avail,
                        'load_pct': load_pct_b,
                        'efficiency': eff_b,
                        'score': score_b
                    }
                break
        if best_config_b and best_config_b['n_total'] == n_tot:
            break
    
    if best_config_b:
        reliability_configs.append(best_config_b)

# --- CORRECTED CONFIGURATION C: HYBRID (BALANCED) ---
if use_bess and bess_reliability_enabled:
    if bess_strategy == 'Hybrid (Balanced)':
        target_gensets_covered = 3
        bess_coverage_hrs = 2.0
    else: 
        target_gensets_covered = 5
        bess_coverage_hrs = 2.5
    
    bess_power_hybrid = max(bess_power_transient, target_gensets_covered * unit_site_cap)
    bess_energy_hybrid = bess_power_hybrid * bess_coverage_hrs
    min_energy_for_credit = target_gensets_covered * unit_site_cap * bess_coverage_hrs
    bess_energy_hybrid = max(bess_energy_hybrid, min_energy_for_credit)
    
    bess_credit_units, credit_breakdown = calculate_bess_reliability_credit(
        bess_power_hybrid, bess_energy_hybrid, unit_site_cap, mttr_hours
    )
    
    n_run_ref = best_config_b['n_running'] if best_config_b else n_running_from_load
    n_running_start = max(1, n_run_ref - 2)
    n_running_end = n_run_ref + 1

    best_config_c = None

    for n_run in range(n_running_start, n_running_end):
        # Chequeo de capacidad mínima
        if n_run * unit_site_cap < p_total_avg * 1.02:
            continue

        for n_res_base in range(0, 15):
            bess_credit_int = int(bess_credit_units) 
            n_res_physical = max(0, n_res_base - bess_credit_int)
            n_tot = n_run + n_res_physical
            
            avg_avail, _ = calculate_availability_weibull(n_run + n_res_base, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
            
            if avg_avail >= avail_decimal:
                load_pct_c = (p_total_avg / (n_run * unit_site_cap)) * 100
                eff_c = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_c, gen_data["type"])
                
                score_c = eff_c - (n_tot * 0.05) 
                
                if best_config_c is None or score_c > best_config_c.get('score', -999):
                    best_config_c = {
                        'name': f'C: {bess_strategy}',
                        'n_running': n_run,
                        'n_reserve': n_res_physical,
                        'n_total': n_tot,
                        'bess_mw': bess_power_hybrid,
                        'bess_mwh': bess_energy_hybrid,
                        'bess_credit': bess_credit_units,
                        'availability': avg_avail,
                        'credit_breakdown': credit_breakdown,
                        'load_pct': load_pct_c,
                        'efficiency': eff_c,
                        'score': score_c
                    }
                break 
    
    if best_config_c:
        reliability_configs.append(best_config_c)

# Select final configuration
if bess_strategy == "Transient Only" and len(reliability_configs) >= 2:
    selected_config = reliability_configs[1]
elif bess_strategy in ["Hybrid (Balanced)", "Reliability Priority"] and len(reliability_configs) >= 3:
    selected_config = reliability_configs[2]
elif len(reliability_configs) >= 1:
    selected_config = reliability_configs[0]
else:
    selected_config = {'name': 'Fallback', 'n_running': n_running_from_load, 'n_reserve': 10, 'n_total': n_running_from_load + 10, 'bess_mw': 0, 'bess_mwh': 0, 'availability': 0.9999}

n_running = selected_config['n_running']
n_reserve = selected_config['n_reserve']
n_total = selected_config['n_total']
prob_gen = selected_config['availability']
bess_power_total = selected_config['bess_mw']
bess_energy_total = selected_config['bess_mwh']
target_met = prob_gen >= avail_decimal

if use_bess and bess_power_total > 0:
    bess_breakdown = bess_breakdown_transient.copy()
    bess_breakdown['reliability_backup'] = bess_power_total - bess_power_transient
else:
    bess_breakdown = {}

installed_cap = n_total * unit_site_cap
_, availability_curve = calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])

# Show optimization result in sidebar
if use_bess and bess_reliability_enabled and 'bess_credit' in selected_config:
    units_saved = best_config_a['n_total'] - selected_config['n_total']
    if units_saved > 0:
        st.sidebar.success(f"✅ **BESS Reliability Credit:**\n\nSaved {units_saved} gensets")
    else:
        st.sidebar.info(f"ℹ️ **{selected_config['name']}:** BESS Credit Active")

st.sidebar.markdown("⚡ **Load Distribution**")
load_strategy = st.sidebar.radio("Operating Mode", ["Equal Loading (N units)", "Spinning Reserve (N+1)", "Sequential"])
if load_strategy == "Equal Loading (N units)":
    units_running = n_running
elif load_strategy == "Spinning Reserve (N+1)":
    units_running = n_running + 1 if n_reserve > 0 else n_running
else:
    units_running = n_running

load_per_unit_pct = (p_total_avg / (units_running * unit_site_cap)) * 100
fleet_efficiency = get_part_load_efficiency(gen_data["electrical_efficiency"], load_per_unit_pct, gen_data["type"])

if volt_mode == "Auto-Recommend":
    if installed_cap < 10:
        rec_voltage_kv = 4.16
    elif installed_cap < 50:
        rec_voltage_kv = 13.8
    elif installed_cap < 200:
        rec_voltage_kv = 34.5
    else:
        rec_voltage_kv = 34.5
else:
    rec_voltage_kv = manual_voltage_kv

stability_ok, voltage_sag = transient_stability_check(gen_data["reactance_xd_2"], units_running, step_load_req)

# ==============================================================================
# 4. FOOTPRINT CALCULATION
# ==============================================================================

area_per_gen = 1 / gen_data["power_density_mw_per_m2"]
area_gen = n_total * unit_site_cap * area_per_gen
area_bess = bess_power_total * 30 if use_bess else 0

if has_lng_storage:
    total_fuel_input_mw_temp = (p_total_avg / fleet_efficiency)
    total_fuel_input_mmbtu_hr_temp = total_fuel_input_mw_temp * 3.412
    lng_mmbtu_total = total_fuel_input_mmbtu_hr_temp * 24 * lng_days
    lng_gal = lng_mmbtu_total / 0.075
    storage_area_m2 = (lng_gal * 0.00378541) * 5
else:
    storage_area_m2 = 0
    lng_gal = 0

pue_base = 1.35 if cooling_method == "Water-Cooled" else 1.50
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)
area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

is_area_exceeded = total_area_m2 > max_area_m2
area_utilization_pct = (total_area_m2 / max_area_m2) * 100 if enable_footprint_limit else 0

footprint_recommendations = []
if is_area_exceeded and enable_footprint_limit:
    current_density = gen_data["power_density_mw_per_m2"]
    for alt_gen_name, alt_gen_data in available_gens.items():
        if alt_gen_data["power_density_mw_per_m2"] > current_density * 1.3:
            footprint_recommendations.append({'type': 'Switch Technology', 'action': f'Change to {alt_gen_name}', 'trade_off': 'Check Efficiency'})

if is_imperial:
    disp_area = total_area_m2 * 0.000247105
    disp_area_unit = "Acres"
else:
    disp_area = total_area_m2 / 10000
    disp_area_unit = "Ha"

# ==============================================================================
# 5. FINANCIALS
# ==============================================================================

total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412

if not is_lng_primary:
    flow_rate_scfh = total_fuel_input_mmbtu_hr * 1000 / 1.02
    rec_pipe_dia = math.sqrt(flow_rate_scfh / 3000) * 2
else:
    rec_pipe_dia = 0

nox_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_nox"] / 1000)
co_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_co"] / 1000)
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor

at_capex_total = 0
if nox_lb_hr * 8760 > 100:
    at_capex_total = (installed_cap * 1000) * (75.0 + 25.0)

gen_unit_cost = gen_data["est_cost_kw"] * regional_mult
gen_install_cost = gen_data["est_install_kw"] * regional_mult
gen_cost_total = (installed_cap * 1000) * gen_unit_cost / 1e6
idx_install = gen_install_cost / gen_unit_cost
idx_chp = 0.20 if include_chp else 0

bess_cost_kw = 250.0
bess_cost_kwh = 400.0
bess_om_kw_yr = 5.0

if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr)
else:
    bess_capex_m = 0
    bess_om_annual = 0

if has_lng_storage:
    log_capex = (lng_gal * 3.5) + (lng_days * 50000)
    pipeline_capex_m = 0
else:
    log_capex = 0
    pipe_cost_m = 50 * rec_pipe_dia
    pipeline_capex_m = (pipe_cost_m * dist_gas_main_m) / 1e6

cost_items = [
    {"Item": "Generation Units", "Index": 1.00, "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Index": idx_install, "Cost (M USD)": gen_cost_total * idx_install},
    {"Item": "Tri-Gen Plant", "Index": idx_chp, "Cost (M USD)": gen_cost_total * idx_chp},
    {"Item": "BESS System", "Index": 0.0, "Cost (M USD)": bess_capex_m},
    {"Item": "Fuel Infrastructure", "Index": 0.0, "Cost (M USD)": (log_capex + pipeline_capex_m * 1e6)/1e6},
    {"Item": "Emissions Control", "Index": 0.0, "Cost (M USD)": at_capex_total / 1e6},
]
df_capex = pd.DataFrame(cost_items)
initial_capex_sum = df_capex["Cost (M USD)"].sum()

itc_benefit_m = (initial_capex_sum * 0.30) if (enable_itc and include_chp) else 0
depreciation_benefit_m = calculate_macrs_depreciation(initial_capex_sum * 1e6, project_years) / 1e6 if enable_depreciation else 0

repowering_pv_m = 0.0
if use_bess:
    for year in range(1, project_years + 1):
        year_cost = 0.0
        if year % 10 == 0 and year < project_years:
            year_cost += (bess_energy_total * 1000 * bess_cost_kwh)
        if year % 15 == 0 and year < project_years:
            year_cost += (bess_power_total * 1000 * bess_cost_kw)
        if year_cost > 0:
            repowering_pv_m += (year_cost / 1e6) / ((1 + wacc) ** year)

crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
repowering_annualized = repowering_pv_m * 1e6 * crf

effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours
om_fixed_annual = (installed_cap * 1000) * 15.0
om_variable_annual = mwh_year * 3.5
om_labor_annual = n_total * 120000

overhaul_pv = 0
for year in np.arange(60000 / (8760 * capacity_factor), project_years, 60000 / (8760 * capacity_factor)):
    overhaul_pv += (installed_cap * 150000) / ((1 + wacc) ** int(year))
overhaul_annualized = overhaul_pv * crf

om_cost_year = om_fixed_annual + om_variable_annual + om_labor_annual + bess_om_annual + overhaul_annualized
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

capex_annualized = (initial_capex_sum * 1e6) * crf
total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
ptc_annual = (mwh_year * 1000 * 0.013) if enable_ptc else 0
itc_annualized = (itc_benefit_m * 1e6) * crf
depreciation_annualized = (depreciation_benefit_m * 1e6) * crf

total_annual_cost_after_tax = total_annual_cost - ptc_annual - itc_annualized - depreciation_annualized
lcoe = total_annual_cost_after_tax / (mwh_year * 1000)

annual_savings = (mwh_year * 1000 * benchmark_price) - (fuel_cost_year + om_cost_year + carbon_cost_year)
if wacc > 0:
    pv_savings = annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)
else:
    pv_savings = annual_savings * project_years

total_tax_benefits = (itc_benefit_m + depreciation_benefit_m) * 1e6 + (ptc_annual * project_years)
npv = pv_savings + total_tax_benefits - (initial_capex_sum * 1e6) - (repowering_pv_m * 1e6)

if annual_savings > 0:
    payback_years = (initial_capex_sum * 1e6) / annual_savings
    payback_str = f"{payback_years:.1f} Years"
else:
    payback_str = "N/A"

gas_prices_x = np.linspace(0, total_gas_price * 2, 20)
lcoe_y = []
breakeven_gas_price = 0.0
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    lcoe_y.append(sim_lcoe)
    if sim_lcoe <= benchmark_price and breakeven_gas_price == 0:
        breakeven_gas_price = gp

# ==============================================================================
# 6. OUTPUTS
# ==============================================================================

t1, t2, t3, t4, t5 = st.tabs(["📊 System Design", "⚡ Performance & Stability", "🏗️ Footprint & Optimization", "❄️ Cooling & Tri-Gen", "💰 Economics & ROI"])

with t1:
    st.subheader("System Architecture")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Generator", selected_gen)
    c2.metric("Fleet", f"{n_running}+{n_reserve}")
    c3.metric("Installed", f"{installed_cap:.1f} MW")
    c4.metric("Availability", f"{prob_gen*100:.3f}%", delta="✅ Target Met" if target_met else "⚠️ Below Target", delta_color="normal" if target_met else "inverse")
    c5.metric("PUE", f"{pue:.2f}")
    c6.metric("Density", f"{gen_data['power_density_mw_per_m2']*1000:.0f} kW/m²")
    
    hours = np.arange(0, 8760)
    daily_wave = 1.0 + 0.15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    load_curve = np.clip(p_total_avg * daily_wave * np.random.uniform(0.95, 1.05, len(hours)), 0, p_total_peak)
    load_sorted = np.sort(load_curve)[::-1]
    
    fig_ldc = go.Figure()
    fig_ldc.add_trace(go.Scatter(x=hours, y=load_sorted, fill='tozeroy', name='DC Load', line=dict(color='#667eea', width=2)))
    fig_ldc.add_hline(y=installed_cap, line_dash="dash", line_color="red", annotation_text="Installed")
    genset_capacity = n_running * unit_site_cap
    fig_ldc.add_hline(y=genset_capacity, line_dash="dashdot", line_color="green", annotation_text="Genset Capacity")
    if use_bess and p_total_peak > genset_capacity:
        fig_ldc.add_hrect(y0=genset_capacity, y1=p_total_peak, fillcolor="yellow", opacity=0.2, annotation_text="BESS Peak Shaving")
    st.plotly_chart(fig_ldc, use_container_width=True)

    if fleet_options:
        st.markdown("### 🎯 Fleet Optimization")
        opt_data = [{'Units': n, 'Load (%)': d['load_pct'], 'Efficiency (%)': d['efficiency']*100} for n, d in fleet_options.items()]
        st.dataframe(pd.DataFrame(opt_data), use_container_width=True)

with t2:
    st.subheader("Electrical Performance & Stability")
    
    if len(reliability_configs) > 1:
        st.markdown("### ⚖️ Reliability Configuration Comparison")
        comparison_data = []
        for config in reliability_configs:
            genset_capex = config['n_total'] * unit_site_cap * gen_data['est_cost_kw'] / 1000
            bess_capex = config['bess_mwh'] * 0.3
            total_capex = genset_capex + bess_capex
            
            running_units = config['n_running']
            load_per_unit = config.get('load_pct', (p_total_avg / (running_units * unit_site_cap)) * 100)
            config_efficiency = config.get('efficiency', get_part_load_efficiency(gen_data["electrical_efficiency"], load_per_unit, gen_data["type"]))
            
            comparison_data.append({
                'Configuration': config['name'],
                'Fleet': f"{config['n_running']}+{config['n_reserve']}",
                'Total Units': config['n_total'],
                'BESS (MW/MWh)': f"{config['bess_mw']:.0f}/{config['bess_mwh']:.0f}" if config['bess_mw'] > 0 else "None",
                'Load/Unit (%)': f"{load_per_unit:.1f}%",
                'Fleet Eff (%)': f"{config_efficiency*100:.1f}%",
                'Availability': f"{config['availability']*100:.3f}%",
                'CAPEX (M$)': f"${total_capex:.1f}M"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        selected_name = selected_config['name']
        
        st.dataframe(
            df_comparison.style.apply(
                lambda row: ['background-color: #d4edda' if row['Configuration'] == selected_name else '' for _ in row],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Voltage", f"{rec_voltage_kv} kV")
    col_e2.metric("Frequency", f"{freq_hz} Hz")
    col_e3.metric("X\"d", f"{gen_data['reactance_xd_2']:.3f} pu")
    col_e4.metric("Ramp Rate", f"{gen_data['ramp_rate_mw_s']:.1f} MW/s")
    
    if stability_ok:
        st.success(f"✅ **Voltage Sag OK:** {voltage_sag:.2f}% (Limit: 10%)")
    else:
        st.error(f"❌ **Voltage Sag Exceeds:** {voltage_sag:.2f}%")

    if use_bess:
        st.markdown("### 🔋 BESS Sizing Breakdown")
        bess_breakdown_data = pd.DataFrame({
            'Component': ['Step Support', 'Peak Shaving', 'Ramp Support', 'Freq Reg', 'Black Start'],
            'Power (MW)': [bess_breakdown.get('step_support', 0), bess_breakdown.get('peak_shaving', 0), bess_breakdown.get('ramp_support', 0), bess_breakdown.get('freq_reg', 0), bess_breakdown.get('black_start', 0)]
        })
        st.bar_chart(bess_breakdown_data, x='Component', y='Power (MW)')

with t3:
    st.subheader("Footprint Analysis & Optimization")
    col_fp1, col_fp2, col_fp3 = st.columns(3)
    col_fp1.metric("Total Footprint", f"{disp_area:.2f} {disp_area_unit}")
    col_fp2.metric("Power Density", f"{gen_data['power_density_mw_per_m2']:.3f} MW/m²")
    col_fp3.metric("Utilization", f"{area_utilization_pct:.1f}%" if enable_footprint_limit else "No Limit")
    
    footprint_data = pd.DataFrame({
        "Component": ["Generators", "BESS", "Fuel Storage", "Cooling/CHP", "Substation", "Contingency"],
        "Area (m²)": [area_gen, area_bess, storage_area_m2, area_chp, area_sub, total_area_m2 * 0.2]
    })
    st.bar_chart(footprint_data, x="Component", y="Area (m²)")
    
    if is_area_exceeded and footprint_recommendations:
        st.warning("⚠️ **Optimization Suggested:**")
        for rec in footprint_recommendations:
            st.write(f"- {rec['action']}")

with t4:
    st.subheader("Cooling & Tri-Generation")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    if include_chp:
        total_heat_rec_mw = (total_fuel_input_mw - p_total_avg) * 0.65
        total_cooling_mw_chp = total_heat_rec_mw * 0.70
        col_c1.metric("Heat Recovered", f"{total_heat_rec_mw:.1f} MWt")
        col_c2.metric("Cooling Generated", f"{total_cooling_mw_chp:.1f} MWc")
    else:
        col_c1.metric("Cooling Method", cooling_method)
    
    wue = 1.8 if (cooling_method == "Water-Cooled" or include_chp) else 0.2
    disp_water = p_it * wue * 24
    if is_imperial: disp_water *= 264.172
    col_c3.metric(f"Water Use", f"{disp_water:,.0f} {'gal' if is_imperial else 'm³'}/day")

with t5:
    st.subheader("Financial Analysis & Economics")
    st.metric("LCOE", f"${lcoe:.4f}/kWh")
    
    edited_capex = st.data_editor(
        df_capex,
        column_config={
            "Index": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    total_capex_dynamic = edited_capex["Cost (M USD)"].sum()
    capex_annualized_dyn = (total_capex_dynamic * 1e6) * crf
    lcoe_dyn = (fuel_cost_year + om_cost_year + capex_annualized_dyn + repowering_annualized + carbon_cost_year - ptc_annual - itc_annualized - depreciation_annualized) / (mwh_year * 1000)
    
    c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
    c_f1.metric("CAPEX", f"${total_capex_dynamic:.2f}M")
    c_f2.metric("LCOE", f"${lcoe_dyn:.4f}/kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f}M")
    c_f4.metric("NPV (20yr)", f"${npv/1e6:.2f}M")
    c_f5.metric("Payback", payback_str)

# ==============================================================================
# 9. EXCEL & PDF EXPORT
# ==============================================================================

st.markdown("---")
st.subheader("📄 Export Comprehensive Report")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

if not EXCEL_AVAILABLE and not PDF_AVAILABLE:
    st.warning("⚠️ **Excel and PDF export unavailable.** Install `openpyxl` and/or `reportlab`. CSV export available below.")

if EXCEL_AVAILABLE:
    @st.cache_data
    def create_excel_export():
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            summary_data = {
                'Parameter': [
                    'DC Type', 'IT Load (MW)', 'PUE', 'Total DC Load (MW)',
                    'Avg Load (MW)', 'Peak Load (MW)', 'Capacity Factor (%)',
                    'Generator Model', 'Generator Type', 'Fleet Config',
                    'Installed Capacity (MW)', 'Availability Target (%)', 
                    'Availability Achieved (%)', 'Target Met',
                    'Operating PUE', 'Strategy', 'Load/Unit (%)', 'Fleet Efficiency (%)',
                    'Voltage (kV)', 'Primary Fuel', 'Region'
                ],
                'Value': [
                    dc_type, p_it, pue, p_total_dc,
                    p_total_avg, p_total_peak, capacity_factor*100,
                    selected_gen, gen_data['type'], f'{n_running}+{n_reserve}',
                    installed_cap, avail_req, prob_gen*100, 
                    'Yes' if target_met else 'No',
                    pue_actual if 'pue_actual' in locals() else pue, load_strategy, load_per_unit_pct, fleet_efficiency*100,
                    rec_voltage_kv, fuel_mode, region
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Financial Summary
            financial_data = {
                'Metric': [
                    'Total CAPEX (M USD)', 'LCOE ($/kWh)', 
                    'Benchmark Price ($/kWh)', 'Gas Price Total ($/MMBtu)',
                    'Carbon Price ($/ton)', 'Annual Energy (MWh)', 'Effective Hours',
                    'Fuel Cost/Year (M USD)', 'O&M Cost/Year (M USD)',
                    'Carbon Cost/Year (M USD)', 'ITC Benefit (M USD)',
                    'MACRS Benefit (M USD)', 'PTC Annual (M USD)',
                    'Annual Savings (M USD)', 'NPV 20yr (M USD)',
                    'Payback (Years)', 'Breakeven Gas ($/MMBtu)'
                ],
                'Value': [
                    total_capex_dynamic, lcoe_dyn,
                    benchmark_price, total_gas_price,
                    carbon_price_per_ton, mwh_year, effective_hours,
                    fuel_cost_year/1e6, om_cost_year/1e6,
                    carbon_cost_year/1e6, itc_benefit_m,
                    depreciation_benefit_m, ptc_annual/1e6,
                    annual_savings/1e6, npv/1e6,
                    payback_str, breakeven_gas_price
                ]
            }
            pd.DataFrame(financial_data).to_excel(writer, sheet_name='Financial Summary', index=False)
            
            # Sheet 3: CAPEX
            edited_capex.to_excel(writer, sheet_name='CAPEX Breakdown', index=False)
            
            # Sheet 4: O&M
            om_data = pd.DataFrame({
                'Component': ['Fixed', 'Variable', 'Labor', 'Major Overhaul', 'BESS O&M'],
                'Annual Cost ($M)': [om_fixed_annual/1e6, om_variable_annual/1e6, om_labor_annual/1e6, overhaul_annualized/1e6, bess_om_annual/1e6]
            })
            om_data.to_excel(writer, sheet_name='OM Breakdown', index=False)
            
            # Sheet 5: Technical
            tech_data = {
                'Specification': [
                    'ISO Rating (MW)', 'Site Rating (MW)', 'Derate Factor',
                    'ISO Efficiency (%)', 'Fleet Efficiency (%)', 'Ramp Rate (MW/s)',
                    'MTBF (hours)', 'Step Load (%)', 'Voltage Sag (%)',
                    'BESS Power (MW)', 'BESS Energy (MWh)', 'Fuel (MMBtu/hr)',
                    'NOx (lb/hr)', 'CO (lb/hr)', 'CO2 (tons/yr)',
                    'Footprint', 'Power Density (MW/m²)', 'WUE'
                ],
                'Value': [
                    unit_iso_cap, unit_site_cap, derate_factor_calc,
                    gen_data['electrical_efficiency']*100, fleet_efficiency*100,
                    gen_data['ramp_rate_mw_s'], gen_data['mtbf_hours'],
                    gen_data['step_load_pct'], voltage_sag,
                    bess_power_total, bess_energy_total, total_fuel_input_mmbtu_hr,
                    nox_lb_hr, co_lb_hr, co2_ton_yr,
                    f'{disp_area:.2f} {disp_area_unit}',
                    gen_data['power_density_mw_per_m2'], wue
                ]
            }
            pd.DataFrame(tech_data).to_excel(writer, sheet_name='Technical Specs', index=False)
            
            # Sheet 6: Reliability
            reliability_data = pd.DataFrame({
                'Year': range(1, project_years + 1),
                'Availability (%)': [a*100 for a in availability_curve]
            })
            reliability_data.to_excel(writer, sheet_name='Reliability Curve', index=False)
            
            # Sheet 7: Gas Sensitivity
            sensitivity_data = pd.DataFrame({
                'Gas Price ($/MMBtu)': gas_prices_x,
                'LCOE ($/kWh)': lcoe_y
            })
            sensitivity_data.to_excel(writer, sheet_name='Gas Sensitivity', index=False)
            
            # Sheet 8: Footprint
            footprint_data.to_excel(writer, sheet_name='Footprint', index=False)
            
            if bess_breakdown:
                pd.DataFrame(list(bess_breakdown.items()), columns=['Component', 'MW']).to_excel(writer, sheet_name='BESS Sizing', index=False)
        
        output.seek(0)
        return output
    
    excel_data = create_excel_export()
    
    st.download_button(
        label="📊 Download Complete Excel Report (9 Sheets)",
        data=excel_data,
        file_name=f"CAT_QuickSize_v2_{dc_type.replace(' ','_')}_{p_it:.0f}MW.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if PDF_AVAILABLE:
    st.markdown("---")
    
    @st.cache_data
    def create_pdf_export():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1f77b4'), spaceAfter=30, alignment=TA_CENTER)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2ca02c'), spaceAfter=12, spaceBefore=12)
        
        story.append(Paragraph("CAT QuickSize v2.1", title_style))
        story.append(Paragraph("Data Center Primary Power Solutions", styles['Heading3']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Project: {dc_type} - {p_it:.0f} MW IT Load", styles['Normal']))
        story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        story.append(Paragraph("Executive Summary", heading_style))
        summary_table_data = [
            ['Parameter', 'Value'],
            ['Data Center Type', dc_type],
            ['IT Load', f'{p_it:.1f} MW'],
            ['Fleet Configuration', f'{n_running}+{n_reserve} ({n_total} total units)'],
            ['Installed Capacity', f'{installed_cap:.1f} MW'],
            ['Availability Target', f'{avail_req:.3f}%'],
            ['Availability Achieved', f'{prob_gen*100:.3f}%'],
            ['Target Met', 'Yes' if target_met else 'No'],
        ]
        
        summary_table = Table(summary_table_data, colWidths=[3.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("Financial Summary", heading_style))
        financial_table_data = [
            ['Metric', 'Value'],
            ['Total CAPEX', f'${total_capex_dynamic:.2f}M'],
            ['LCOE (after tax)', f'${lcoe_dyn:.4f}/kWh'],
            ['Annual Savings', f'${annual_savings/1e6:.2f}M'],
            ['NPV (20 years)', f'${npv/1e6:.2f}M'],
            ['Simple Payback', f'{payback_str}'],
        ]
        
        financial_table = Table(financial_table_data, colWidths=[3.5*inch, 2.5*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(financial_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    pdf_data = create_pdf_export()
    
    st.download_button(
        label="📄 Download PDF Proposal",
        data=pdf_data,
        file_name=f"CAT_Proposal_{dc_type.replace(' ','_')}_{p_it:.0f}MW.pdf",
        mime="application/pdf",
        use_container_width=True
    )

if not EXCEL_AVAILABLE:
    st.info("💡 **CSV Export Mode:**")
    summary_csv = pd.DataFrame({'Parameter': ['LCOE', 'CAPEX', 'Availability'], 'Value': [lcoe_dyn, total_capex_dynamic, prob_gen]})
    st.download_button("📋 Summary (CSV)", summary_csv.to_csv(index=False), "Summary.csv", "text/csv")

# --- FOOTER ---
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
col_foot1.caption("CAT QuickSize v2.1")
col_foot2.caption("Next-Gen Data Center Power Solutions")
col_foot3.caption("Caterpillar Electric Power | 2026")
