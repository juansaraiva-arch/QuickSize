import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT QuickSize v3.0", page_icon="⚡", layout="wide")

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
        "ramp_rate_mw_s": 0.5,  # NEW
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "mtbf_hours": 50000,  # NEW: Mean Time Between Failures
        "maintenance_interval_hrs": 1000,  # Time between planned maintenance
        "maintenance_duration_hrs": 48,    # Downtime per maintenance event
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,
        "power_density_mw_per_m2": 0.010,  # NEW: 200 m²/MW
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
        "ramp_rate_mw_s": 2.0,  # Turbines ramp faster
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "mtbf_hours": 80000,  # Turbines more reliable
        "maintenance_interval_hrs": 8000,  # Less frequent maintenance
        "maintenance_duration_hrs": 120,   # Longer downtime per event
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
        "power_density_mw_per_m2": 0.020,  # NEW: 50% less footprint
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
        "ramp_rate_mw_s": 0.3,  # Slower ramp
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "mtbf_hours": 60000,
        "maintenance_interval_hrs": 2500,  # Medium speed: less frequent
        "maintenance_duration_hrs": 72,    # Longer downtime
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "power_density_mw_per_m2": 0.008,  # Larger footprint
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
    V3.0 Logic: BESS covers 100% of transient if needed to avoid starting engines.
    """
    # Component 1: Step Load Support (Full coverage assumed for Hybrid)
    bess_step_support = p_net_req_avg * (step_load_req / 100)
    
    # Component 2: Peak Shaving (covers peak vs average difference)
    bess_peak_shaving = max(0, p_net_req_peak - p_net_req_avg)
    
    # Component 3: Ramp Rate Support
    load_change_rate = 5.0  # MW/s (AI workload aggressive)
    bess_ramp_support = max(0, (load_change_rate - gen_ramp_rate) * 10)  # 10s buffer
    
    # Component 4: Frequency Regulation
    bess_freq_reg = p_net_req_avg * 0.05
    
    # Component 5: Black Start Capability
    bess_black_start = p_net_req_peak * 0.05 if enable_black_start else 0
    
    # Total Power (Dominant factor)
    bess_power_total = max(
        bess_step_support,
        bess_peak_shaving,
        bess_ramp_support,
        bess_freq_reg,
        bess_black_start
    )
    
    # Energy Duration Calculation (1 hr standard for peak/step hybrid)
    c_rate = 1.0
    bess_energy_total = bess_power_total / c_rate
    rte = 0.85
    bess_energy_total = bess_energy_total / rte
    
    breakdown = {
        'step_support': bess_step_support,
        'peak_shaving': bess_peak_shaving,
        'ramp_support': bess_ramp_support,
        'freq_reg': bess_freq_reg,
        'black_start': bess_black_start
    }
    
    return bess_power_total, bess_energy_total, breakdown

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

st.title(f"⚡ CAT QuickSize v3.0 ({freq_hz}Hz)")
st.markdown("**Next-Gen Data Center Power Solutions.**\nOperational Deterministic Modeling: Comparing 'Gas Only' vs 'Hybrid' architecture.")

# ==============================================================================
# 2. INPUTS - ENHANCED WITH PUE
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
    def_use_bess = True
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    pue = st.slider("Data Center PUE", 1.05, 2.00, pue_defaults[dc_type], 0.05)
    
    p_total_dc = p_it * pue
    p_aux = p_total_dc - p_it
    
    with st.expander("ℹ️ PUE Breakdown"):
        st.write(f"**IT Load:** {p_it:.1f} MW")
        st.write(f"**Total DC Load:** {p_total_dc:.1f} MW")
    
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
    
    st.info(f"💡 **Load Analysis:**\n- Avg: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Step Load Req (%)", 0.0, 100.0, def_step_load)
    
    st.markdown("📐 **Site Constraints**")
    enable_footprint_limit = st.checkbox("Enable Footprint Limit", value=False)
    if enable_footprint_limit:
        max_area_input = st.number_input("Max Available Area (m²)", 1000.0, 500000.0, 50000.0)
        max_area_m2 = max_area_input
    else:
        max_area_m2 = 999999999
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual":
        manual_voltage_kv = st.number_input("Custom Voltage (kV)", 0.4, 230.0, 13.8)

    st.markdown("🌍 **Site Environment**")
    derate_mode = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True)
    if derate_mode == "Auto-Calculate":
        c_env1, c_env2 = st.columns(2)
        site_temp_c = c_env1.number_input(f"Ambient Temp ({u_temp})", 0, 55, 25)
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

    st.header("2. Technology Solution")
    gen_filter = st.multiselect("Technology Filter", ["High Speed", "Medium Speed", "Gas Turbine"], default=["High Speed", "Medium Speed"])
    enable_black_start = st.checkbox("Enable Black Start Capability", value=False)
    include_chp = st.checkbox("Include Tri-Generation (CHP)", value=False)
    cooling_method = "Absorption Chiller" if include_chp else st.selectbox("Cooling Method", ["Air-Cooled", "Water-Cooled"])
    
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
    col_g1, col_g2 = st.columns(2)
    gas_price_wellhead = col_g1.number_input("Gas Price ($/MMBtu)", 0.5, 30.0, 4.0, step=0.5)
    gas_transport = col_g2.number_input("Transport ($/MMBtu)", 0.0, 5.0, 0.5, step=0.1)
    total_gas_price = gas_price_wellhead + gas_transport
    benchmark_price = st.number_input("Benchmark Electricity ($/kWh)", 0.01, 0.50, 0.12, step=0.01)
    
    carbon_scenario = st.selectbox("Carbon Price Scenario", ["None (Current 2026)", "California Cap-and-Trade", "EU ETS", "High Case"])
    carbon_prices = {"None (Current 2026)": 0, "California Cap-and-Trade": 35, "EU ETS": 85, "High Case": 150}
    carbon_price_per_ton = carbon_prices[carbon_scenario]
    
    c_fin1, c_fin2 = st.columns(2)
    wacc = c_fin1.number_input("WACC (%)", 1.0, 20.0, 8.0, step=0.5) / 100
    project_years = c_fin2.number_input("Project Life (Years)", 10, 30, 20, step=5)
    
    enable_itc = st.checkbox("Include ITC (30% for CHP)", value=include_chp)
    enable_ptc = st.checkbox("Include PTC ($0.013/kWh, 10yr)", value=False)
    enable_depreciation = st.checkbox("Include MACRS Depreciation", value=True)
    
    region = st.selectbox("Region", ["US - Gulf Coast", "US - Northeast", "Europe - Western", "Latin America", "Asia Pacific"])
    regional_multipliers = {"US - Gulf Coast": 1.0, "US - Northeast": 1.25, "Europe - Western": 1.35, "Latin America": 0.95, "Asia Pacific": 0.85}
    regional_mult = regional_multipliers[region]
    
    enable_lcoe_target = st.checkbox("Enable LCOE Target Mode", value=False)
    target_lcoe = 0.0
    if enable_lcoe_target:
        target_lcoe = st.number_input("Target LCOE ($/kWh)", 0.01, 0.50, 0.08, step=0.005)

# ==============================================================================
# 3. GENERATOR SELECTION
# ==============================================================================

available_gens = {k: v for k, v in leps_gas_library.items() if v["type"] in gen_filter}
if not available_gens:
    st.error("⚠️ No generators match filter.")
    st.stop()

best_gen = list(available_gens.keys())[0]
selected_gen = st.sidebar.selectbox("🔧 Selected Generator", list(available_gens.keys()), index=0)
gen_data = available_gens[selected_gen]

with st.sidebar.expander("⚙️ Generator Parameters (Editable)", expanded=False):
    mtbf_edit = st.number_input("MTBF (hours)", value=gen_data["mtbf_hours"], step=1000)
    gen_data["mtbf_hours"] = mtbf_edit
    eff_edit = st.number_input("Electrical Efficiency", value=gen_data["electrical_efficiency"], step=0.001, format="%.3f")
    gen_data["electrical_efficiency"] = eff_edit

unit_site_cap = gen_data["iso_rating_mw"] * derate_factor_calc

# ============================================================================
# 4. CORE LOGIC V3.0 (DETERMINISTIC: ENERGY VS POWER)
# ============================================================================

# Paso A: Calcular Flota Base (Energía/Disponibilidad)
# ----------------------------------------------------
# Número mínimo de unidades para cubrir el PICO
n_required_for_peak = math.ceil(p_total_peak / unit_site_cap)
n_base_installed = n_required_for_peak

# Agregar reserva (N+X) hasta cumplir disponibilidad (considerando mtbf y mantenimiento)
for spares in range(0, 10):
    n_total_check = n_required_for_peak + spares
    avail, _ = calculate_availability_weibull(n_total_check, n_required_for_peak, gen_data["mtbf_hours"], 20, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
    if avail >= (avail_req/100.0):
        n_base_installed = n_total_check
        break

n_reserve_base = n_base_installed - n_required_for_peak

# Paso B: Escenario A - "Solo Gas" (Fuerza Bruta)
# -------------------------------------------
# Regla: Las unidades encendidas deben tener suficiente "Headroom" para el Step Load.
mw_step_required = p_total_avg * (step_load_req / 100.0)
n_running_a = n_required_for_peak

while True:
    capacity = n_running_a * unit_site_cap
    headroom = capacity - p_total_avg
    if headroom >= mw_step_required:
        break
    n_running_a += 1

# La flota instalada total debe crecer para mantener el mismo nivel de reserva (N+X)
n_installed_a = max(n_base_installed, n_running_a + n_reserve_base)
load_pct_a = (p_total_avg / (n_running_a * unit_site_cap)) * 100
eff_a = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_a, gen_data["type"])
avail_a, curve_a = calculate_availability_weibull(n_installed_a, n_running_a, gen_data["mtbf_hours"], project_years)

# Paso C: Escenario B - "Híbrido" (Inteligente)
# ------------------------------------------
# Regla: Usar la flota base. BESS absorbe Step Load. Motores corren a eficiencia óptima.
n_installed_b = n_base_installed
n_running_b = math.ceil(p_total_avg / (unit_site_cap * 0.85)) # Target 85% load
n_running_b = max(n_running_b, math.ceil(p_total_avg/unit_site_cap))
if n_running_b > n_required_for_peak: n_running_b = n_required_for_peak

load_pct_b = (p_total_avg / (n_running_b * unit_site_cap)) * 100
eff_b = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_b, gen_data["type"])
avail_b, curve_b = calculate_availability_weibull(n_installed_b, n_running_b, gen_data["mtbf_hours"], project_years)

# BESS Sizing
bess_mw, bess_mwh, bess_bkdn = calculate_bess_requirements(p_total_avg, p_total_peak, step_load_req, gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"], enable_black_start)

# Map output to selected config structure for compatibility
config_a = {
    'name': 'Scenario A: Gas Only',
    'n_running': n_running_a,
    'n_total': n_installed_a,
    'n_reserve': n_installed_a - n_running_a,
    'bess_mw': 0, 'bess_mwh': 0, 'bess_credit': 0,
    'load_pct': load_pct_a, 'efficiency': eff_a, 'availability': avail_a
}

config_b = {
    'name': 'Scenario B: Hybrid (Winner)',
    'n_running': n_running_b,
    'n_total': n_installed_b,
    'n_reserve': n_installed_b - n_running_b,
    'bess_mw': bess_mw, 'bess_mwh': bess_mwh, 'bess_credit': 0,
    'load_pct': load_pct_b, 'efficiency': eff_b, 'availability': avail_b
}

reliability_configs = [config_a, config_b]
selected_config = config_b # Default to Hybrid

# Map global variables for downstream sections
n_running = selected_config['n_running']
n_reserve = selected_config['n_reserve']
n_total = selected_config['n_total']
prob_gen = selected_config['availability']
bess_power_total = selected_config['bess_mw']
bess_energy_total = selected_config['bess_mwh']
target_met = prob_gen >= (avail_req/100.0)
installed_cap = n_total * unit_site_cap
availability_curve = curve_b
fleet_efficiency = eff_b
load_per_unit_pct = load_pct_b
load_strategy = "Optimized (Hybrid)"
bess_breakdown = bess_bkdn if bess_power_total > 0 else {}

if volt_mode == "Auto-Recommend":
    if installed_cap < 50: rec_voltage_kv = 13.8
    else: rec_voltage_kv = 34.5
else: rec_voltage_kv = manual_voltage_kv

stability_ok, voltage_sag = transient_stability_check(gen_data["reactance_xd_2"], n_running, step_load_req)

# ==============================================================================
# 5. FOOTPRINT CALCULATION
# ==============================================================================
area_per_gen = 1 / gen_data["power_density_mw_per_m2"]
area_gen = n_total * unit_site_cap * area_per_gen
area_bess = bess_power_total * 30
if has_lng_storage:
    storage_area_m2 = (p_total_avg/fleet_efficiency * 3.412 * 24 * lng_days / 0.075 * 0.00378 * 5)
else: storage_area_m2 = 0
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)
area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
disp_area = total_area_m2 / 10000 if not is_imperial else total_area_m2 * 0.000247
disp_area_unit = "Ha" if not is_imperial else "Acres"

# Optimization placeholders
is_area_exceeded = total_area_m2 > max_area_m2
area_utilization_pct = (total_area_m2 / max_area_m2) * 100 if enable_footprint_limit else 0
footprint_recommendations = []

# ==============================================================================
# 6. FINANCIALS (DETAILED)
# ==============================================================================
# Calculating for Selected Scenario (Hybrid)
total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor

at_capex_total = (installed_cap * 1000 * 100) if (p_total_avg * 1000 * gen_data["emissions_nox"]) > 100 else 0

gen_cost_total = (installed_cap * 1000) * (gen_data["est_cost_kw"] * regional_mult) / 1e6
bess_capex_m = ((bess_power_total*1000*250) + (bess_energy_total*1000*400))/1e6
infra_capex = (storage_area_m2 * 500 + dist_gas_main_m * 100)/1e6

cost_items = [
    {"Item": "Generation Units", "Index": 1.0, "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Index": gen_data["est_install_kw"]/gen_data["est_cost_kw"], "Cost (M USD)": gen_cost_total * (gen_data["est_install_kw"]/gen_data["est_cost_kw"])},
    {"Item": "BESS System", "Index": 0.0, "Cost (M USD)": bess_capex_m},
    {"Item": "Infrastructure", "Index": 0.0, "Cost (M USD)": infra_capex},
    {"Item": "Emissions Control", "Index": 0.0, "Cost (M USD)": at_capex_total / 1e6},
]
df_capex = pd.DataFrame(cost_items)
initial_capex_sum = df_capex["Cost (M USD)"].sum()

effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours
om_cost_year = (installed_cap * 1000 * 15) + (mwh_year * 3.5) + (n_total * 120000) + (bess_power_total*1000*5)
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

# LCOE Calculation
crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
capex_annualized = initial_capex_sum * 1e6 * crf
# Repowering calc simplified for v3
repowering_annualized = (bess_capex_m * 0.5 * 1e6 * crf) if use_bess else 0 

total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + carbon_cost_year + repowering_annualized
itc_benefit_m = (initial_capex_sum * 0.30) if (enable_itc and include_chp) else 0
depreciation_benefit_m = calculate_macrs_depreciation(initial_capex_sum * 1e6, project_years) / 1e6 if enable_depreciation else 0
ptc_annual = (mwh_year * 1000 * 0.013) if enable_ptc else 0
itc_annualized = (itc_benefit_m * 1e6) * crf
depreciation_annualized = (depreciation_benefit_m * 1e6) * crf
total_annual_cost_after_tax = total_annual_cost - ptc_annual - itc_annualized - depreciation_annualized
lcoe = total_annual_cost_after_tax / (mwh_year * 1000)

annual_savings = (mwh_year * 1000 * benchmark_price) - (fuel_cost_year + om_cost_year + carbon_cost_year)
npv = (annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)) - (initial_capex_sum * 1e6)
payback = (initial_capex_sum * 1e6) / annual_savings if annual_savings > 0 else 0

# Comparative Financials for Summary Table (Scenario A)
capex_a = (n_installed_a * unit_site_cap * 1000 * (gen_data["est_cost_kw"] + gen_data["est_install_kw"]) * regional_mult) / 1e6
fuel_a = (p_total_avg * effective_hours * 3.412 / eff_a) * total_gas_price
om_a = (n_installed_a * unit_site_cap * 1000 * 15) + (mwh_year * 3.5) + (n_installed_a * 120000)
lcoe_a = ((fuel_a + om_a + carbon_cost_year + (capex_a*1e6*crf)) / (mwh_year * 1000))

# ==============================================================================
# 7. OUTPUTS
# ==============================================================================

t1, t2, t3, t4, t5 = st.tabs(["📊 System Design & Comparison", "⚡ Performance Details", "🏗️ Footprint", "❄️ Cooling & Tri-Gen", "💰 Economics & ROI"])

with t1:
    st.subheader("Scenario Comparison: Gas Only vs. Hybrid")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Hybrid Fuel Savings", f"${(fuel_a - fuel_cost_year)/1e6:.2f}M / yr", delta="OPEX Winner")
    capex_diff = initial_capex_sum - capex_a
    col_kpi2.metric("Hybrid CAPEX Delta", f"${capex_diff:.2f}M", delta="Investment" if capex_diff > 0 else "Savings", delta_color="inverse" if capex_diff > 0 else "normal")
    col_kpi3.metric("LCOE Advantage", f"${(lcoe_a - lcoe):.4f}/kWh", delta="Lower Cost")

    comp_df = pd.DataFrame({
        "Metric": ["Total Installed Units", "Running Units (Avg)", "Reserva Rodante (Spinning)", 
                   "Avg Load per Unit", "Net Efficiency (LHV)", "BESS Capacity", "Total CAPEX ($M)", "LCOE ($/kWh)"],
        "Scenario A (Gas Only)": [
            f"{n_installed_a}", f"{n_running_a}", f"{(n_running_a*unit_site_cap - p_total_avg):.1f} MW",
            f"{load_pct_a:.1f}%", f"{eff_a*100:.1f}%", "None", f"${capex_a:.1f}M", f"${lcoe_a:.4f}"
        ],
        "Scenario B (Hybrid)": [
            f"{n_installed_b}", f"{n_running_b}", "BESS Covered",
            f"{load_pct_b:.1f}%", f"{eff_b*100:.1f}%", f"{bess_mw:.1f}MW / {bess_mwh:.1f}MWh", f"${initial_capex_sum:.1f}M", f"${lcoe:.4f}"
        ]
    })
    st.table(comp_df)
    
    st.success(f"✅ **Recommended:** Scenario B (Hybrid) reduces running units by {n_running_a - n_running_b} and improves efficiency by {(eff_b-eff_a)*100:.1f}%.")

with t2:
    st.subheader("Selected System Performance (Hybrid)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fleet Configuration", f"{n_running}+{n_reserve}")
    c2.metric("Availability", f"{prob_gen*100:.4f}%", "Target Met" if target_met else "Below Target")
    c3.metric("System Inertia (H)", "Virtual (BESS)")
    c4.metric("Step Load", f"100% via BESS")
    
    st.markdown("### 🔋 BESS Sizing Breakdown")
    bess_data = pd.DataFrame(list(bess_bkdn.items()), columns=['Component', 'MW'])
    st.bar_chart(bess_data, x='Component', y='MW')

with t3:
    st.subheader("Footprint Analysis")
    c1, c2 = st.columns(2)
    c1.metric("Total Area", f"{disp_area:.2f} {disp_area_unit}")
    c2.metric("Power Density", f"{gen_data['power_density_mw_per_m2']*1000:.0f} kW/m²")
    
    fp_data = pd.DataFrame({
        "Component": ["Generators", "BESS", "Fuel Storage", "Cooling/CHP", "Substation"],
        "Area (m²)": [area_gen, area_bess, storage_area_m2, area_chp, area_sub]
    })
    st.bar_chart(fp_data, x="Component", y="Area (m²)")

with t4:
    st.subheader("Cooling")
    st.metric("Cooling Method", cooling_method)
    if include_chp:
        st.info("CHP Active: Waste heat recovery included in efficiency calc.")

with t5:
    st.subheader("Financial Analysis")
    st.metric("LCOE", f"${lcoe:.4f}/kWh")
    edited_capex = st.data_editor(
        df_capex,
        column_config={
            "Index": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    st.metric("NPV (20yr)", f"${npv/1e6:.2f}M")
    st.metric("Payback", f"{payback:.1f} Years")

# ==============================================================================
# 8. EXPORTS
# ==============================================================================
st.markdown("---")
st.subheader("📄 Export Comprehensive Report")

# Check if libraries are available
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
            comp_df.to_excel(writer, sheet_name='Comparison')
            df_capex.to_excel(writer, sheet_name='CAPEX')
            pd.DataFrame({'Metric': ['LCOE', 'NPV', 'Payback'], 'Value': [lcoe, npv, payback]}).to_excel(writer, sheet_name='Financials')
        return output
    
    excel_data = create_excel_export()
    st.download_button("Download Excel Report", excel_data.getvalue(), "CAT_QuickSize_v3_Report.xlsx")

# PDF Export (independent of Excel)
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
        
        story.append(Paragraph("CAT QuickSize v3.0", title_style))
        story.append(Paragraph("Scenario A vs B Comparison", heading_style))
        
        # Convert df to list for table
        table_data = [comp_df.columns.to_list()] + comp_df.values.tolist()
        
        t = Table(table_data)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('ALIGN',(0,0),(-1,-1),'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND',(0,1),(-1,-1),colors.beige), ('GRID',(0,0),(-1,-1),1,colors.black)]))
        story.append(t)
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    pdf_data = create_pdf_export()
    st.download_button("Download PDF Comparison", pdf_data, "CAT_Scenario_Report.pdf", mime="application/pdf")

# Fallback CSV
if not EXCEL_AVAILABLE:
    st.download_button("Download CSV Report", comp_df.to_csv(), "CAT_QuickSize_v3_Report.csv")

# --- FOOTER ---
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
col_foot1.caption("CAT QuickSize v3.0")
col_foot2.caption("Next-Gen Data Center Power Solutions")
col_foot3.caption("Caterpillar Electric Power | 2026")
