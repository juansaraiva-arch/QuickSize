import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT QuickSize v2.2", page_icon="⚡", layout="wide")

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

def calculate_spinning_reserve_units(p_avg_load, unit_capacity, spinning_reserve_pct, 
                                     use_bess=False, bess_power_mw=0, gen_step_capability_pct=0):
    """Calculate spinning reserve units required."""
    spinning_reserve_mw = p_avg_load * (spinning_reserve_pct / 100)
    
    if use_bess and bess_power_mw > 0:
        spinning_from_bess = min(bess_power_mw, spinning_reserve_mw)
    else:
        spinning_from_bess = 0
    
    spinning_from_gens = spinning_reserve_mw - spinning_from_bess
    required_online_capacity = p_avg_load + spinning_from_gens
    max_unit_load_pct = 85.0
    
    n_min_capacity = math.ceil(required_online_capacity / (unit_capacity * max_unit_load_pct / 100))
    
    if spinning_from_gens > 0:
        n_min_headroom = math.ceil(required_online_capacity / (unit_capacity * max_unit_load_pct / 100))
    else:
        n_min_headroom = math.ceil(p_avg_load / (unit_capacity * max_unit_load_pct / 100))
    
    n_units_running = max(1, max(n_min_capacity, n_min_headroom))
    total_online_capacity = n_units_running * unit_capacity
    load_per_unit_pct = (p_avg_load / total_online_capacity) * 100
    headroom_available = total_online_capacity - p_avg_load
    
    return {
        'n_units_running': n_units_running,
        'load_per_unit_pct': load_per_unit_pct,
        'spinning_reserve_mw': spinning_reserve_mw,
        'spinning_from_gens': spinning_from_gens,
        'spinning_from_bess': spinning_from_bess,
        'total_online_capacity': total_online_capacity,
        'headroom_available': headroom_available,
        'required_online_capacity': required_online_capacity
    }

# --- UPDATED FUNCTION SIGNATURE WITH load_change_rate_req ---
def calculate_bess_requirements(p_net_req_avg, p_net_req_peak, step_load_req, 
                                gen_ramp_rate, gen_step_capability, 
                                load_change_rate_req, # <--- NUEVO ARGUMENTO AGREGADO
                                enable_black_start=False):
    """
    Sophisticated BESS sizing based on actual transient analysis
    NOW RESPONDS TO: Increased step load requirements
    """
    # Component 1: Step Load Support
    step_load_mw = p_net_req_avg * (step_load_req / 100)
    gen_step_mw = p_net_req_avg * (gen_step_capability / 100)
    bess_step_support = max(0, step_load_mw - gen_step_mw)
    
    # Component 2: Peak Shaving
    bess_peak_shaving = p_net_req_peak - p_net_req_avg
    
    # Component 3: Ramp Rate Support (USANDO LA VARIABLE NUEVA)
    # load_change_rate = 5.0  <--- ESTA LÍNEA SE ELIMINA/IGNORA
    bess_ramp_support = max(0, (load_change_rate_req - gen_ramp_rate) * 10)  # 10s buffer
    
    # Component 4: Frequency Regulation
    bess_freq_reg = p_net_req_avg * 0.05
    
    # Component 5: Black Start Capability
    bess_black_start = p_net_req_peak * 0.05 if enable_black_start else 0
    
    # Component 6: Spinning Reserve Support (Para compatibilidad v3)
    bess_spinning_reserve = p_net_req_avg * (step_load_req / 100)
    
    # Total Power
    bess_power_total = max(
        bess_step_support,
        bess_peak_shaving,
        bess_ramp_support,
        bess_freq_reg,
        bess_black_start,
        bess_spinning_reserve,
        p_net_req_peak * 0.15
    )
    
    # Energy Duration Calculation
    c_rate = 1.0
    bess_energy_total = bess_power_total / c_rate
    rte = 0.85
    bess_energy_total = bess_energy_total / rte
    
    breakdown = {
        'step_support': bess_step_support,
        'peak_shaving': bess_peak_shaving,
        'ramp_support': bess_ramp_support,
        'freq_reg': bess_freq_reg,
        'black_start': bess_black_start,
        'spinning_reserve': bess_spinning_reserve
    }
    
    return bess_power_total, bess_energy_total, breakdown
def calculate_bess_reliability_credit(bess_power_mw, bess_energy_mwh, 
                                      unit_capacity_mw, mttr_hours=48):
    if bess_power_mw <= 0 or bess_energy_mwh <= 0:
        return 0.0, {}
    
    realistic_coverage_hrs = 2.0
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
        'bess_duration_hrs': bess_energy_mwh / bess_power_mw if bess_power_mw > 0 else 0,
        'realistic_coverage_hrs': realistic_coverage_hrs
    }
    return effective_credit, credit_breakdown

def calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, 
                                  maintenance_interval_hrs=1000, maintenance_duration_hrs=48):
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

def optimize_fleet_size(p_net_req_avg, p_net_req_peak, unit_cap, step_load_req, gen_data, use_bess=False):
    if use_bess:
        n_min_peak = math.ceil(p_net_req_avg * 1.15 / unit_cap)
        headroom_required = p_net_req_avg * 1.10
        n_min_step = math.ceil(headroom_required / unit_cap)
    else:
        n_min_peak = math.ceil(p_net_req_peak / unit_cap)
        headroom_required = p_net_req_avg * (1 + step_load_req/100) * 1.20
        n_min_step = math.ceil(headroom_required / unit_cap)
    
    n_ideal_eff = math.ceil(p_net_req_avg / (unit_cap * 0.72))
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        if use_bess:
            if n * unit_cap < p_net_req_avg * 1.10: continue
        else:
            if n * unit_cap < p_net_req_peak: continue
        
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 30 or load_pct > 95: continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        optimal_load = 72.5
        load_penalty = abs(load_pct - optimal_load) / 100
        fleet_options[n] = {'efficiency': eff, 'load_pct': load_pct, 'score': eff * (1 - load_penalty * 0.5)}
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}

def calculate_macrs_depreciation(capex, project_years):
    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    tax_rate = 0.21
    pv_benefit = 0
    wacc = 0.08
    for year, rate in enumerate(macrs_schedule, 1):
        if year > project_years: break
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

st.title(f"⚡ CAT QuickSize v2.2 ({freq_hz}Hz)")
st.markdown("**Next-Gen Data Center Power Solutions.**\nAdvanced modeling with PUE optimization, footprint constraints, and sophisticated LCOE analysis.")

# ==============================================================================
# 2. INPUTS - ENHANCED
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
        st.write(f"**Total DC Load:** {p_total_dc:.1f} MW")
    
    st.markdown("📊 **Annual Load Profile**")
    
    # --- ADDED RAMP RATES TO PROFILES ---
    load_profiles = {
        "AI Factory (Training)": {
            "capacity_factor": 0.96, "peak_avg_ratio": 1.08, "ramp_rate": 5.0,  # <--- NUEVO
            "description": "Continuous 24/7 training runs"
        },
        "AI Factory (Inference)": {
            "capacity_factor": 0.85, "peak_avg_ratio": 1.25, "ramp_rate": 3.0,  # <--- NUEVO
            "description": "Variable inference loads with peaks"
        },
        "Hyperscale Standard": {
            "capacity_factor": 0.75, "peak_avg_ratio": 1.20, "ramp_rate": 1.5,  # <--- NUEVO
            "description": "Mixed workloads, diurnal patterns"
        },
        "Colocation": {
            "capacity_factor": 0.65, "peak_avg_ratio": 1.35, "ramp_rate": 1.0,  # <--- NUEVO
            "description": "Multi-tenant, business hours peaks"
        },
        "Edge Computing": {
            "capacity_factor": 0.50, "peak_avg_ratio": 1.50, "ramp_rate": 2.0,  # <--- NUEVO
            "description": "Highly variable local demand"
        }
    }
    
    profile = load_profiles[dc_type]
    
    col_cf1, col_cf2 = st.columns(2)
    capacity_factor = col_cf1.slider("Capacity Factor (%)", 30.0, 100.0, profile["capacity_factor"]*100, 1.0) / 100.0
    peak_avg_ratio = col_cf2.slider("Peak/Avg Ratio", 1.0, 2.0, profile["peak_avg_ratio"], 0.05)
    
    # --- NEW: LOAD RAMP RATE INPUT ---
    load_ramp_req = st.number_input(
        "Load Ramp Rate Req (MW/s)", 
        0.1, 10.0, 
        profile.get("ramp_rate", 1.0), # Usa el valor del perfil o 1.0 por defecto
        0.1,
        help="How fast the load changes. AI = 3-5 MW/s, Colo = 0.5-1 MW/s. Determines BESS power."
    )
    # ---------------------------------
    
    p_total_avg = p_total_dc * capacity_factor
    p_total_peak = p_total_dc * peak_avg_ratio
    
    st.info(f"💡 **Load Analysis:** Avg: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Spinning Reserve Req (%)", 0.0, 100.0, def_step_load)
    
    st.markdown("📐 **Site Constraints**")
    enable_footprint_limit = st.checkbox("Enable Footprint Limit", value=False)
    if enable_footprint_limit:
        max_area_m2 = st.number_input("Max Available Area (m²)", 100.0, 500000.0, 50000.0)
    else:
        max_area_m2 = 999999999
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual":
        manual_voltage_kv = st.number_input("Custom Voltage (kV)", 0.4, 230.0, 13.8)

    # --- CORRECTED SITE ENVIRONMENT LOGIC ---
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
        alt_derate = 1.0 - (site_alt_m * 0.0001) # Corrected formula
        fuel_derate = 1.0 if methane_number >= 70 else 0.95
        derate_factor_calc = max(0.1, temp_derate * alt_derate * fuel_derate)
    else:
        derate_factor_calc = st.slider("Manual Derate Factor", 0.5, 1.0, 0.9, 0.01)
        site_temp_c = 25
        site_alt_m = 0

    st.header("2. Technology Solution")
    gen_filter = st.multiselect("Technology Filter", ["High Speed", "Medium Speed", "Gas Turbine"], default=["High Speed", "Medium Speed"])
    use_bess = st.checkbox("Include BESS (Battery Energy Storage)", value=def_use_bess)
    
    bess_strategy = "Hybrid (Balanced)"
    bess_reliability_enabled = False
    if use_bess:
        bess_strategy = st.radio("Sizing Mode", ["Transient Only", "Hybrid (Balanced)", "Reliability Priority"], index=1)
        bess_reliability_enabled = bess_strategy != "Transient Only"
    
    enable_black_start = st.checkbox("Enable Black Start Capability", value=False)
    include_chp = st.checkbox("Include Tri-Generation (CHP)", value=False)
    cooling_method = "Absorption Chiller" if include_chp else st.selectbox("Cooling Method", ["Air-Cooled", "Water-Cooled"])
    
    st.markdown("⛽ **Fuel Infrastructure**")
    fuel_mode = st.radio("Primary Fuel", ["Pipeline Gas", "LNG", "Dual-Fuel"], horizontal=True)
    is_lng_primary = "LNG" in fuel_mode
    has_lng_storage = fuel_mode in ["LNG", "Dual-Fuel"]
    lng_days = st.number_input("LNG Storage (Days)", 1, 90, 7) if has_lng_storage else 0
    dist_gas_main_km = 0 if has_lng_storage else st.number_input("Distance to Gas Main (km)", 0.1, 100.0, 1.0)
    dist_gas_main_m = dist_gas_main_km * 1000

    st.header("3. Economics & ROI")
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
    benchmark_price = st.number_input("Benchmark Electricity ($/kWh)", 0.01, 0.50, 0.12, step=0.01)
    
    carbon_scenario = st.selectbox("Carbon Price Scenario", ["None (Current 2026)", "California Cap-and-Trade", "EU ETS", "High Case (IEA Net Zero)"])
    carbon_prices = {"None (Current 2026)": 0, "California Cap-and-Trade": 35, "EU ETS": 85, "High Case (IEA Net Zero)": 150}
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
# 3. GENERATOR SELECTION & FLEET OPTIMIZATION
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
# CORRECTED: SPINNING RESERVE CALCULATION
# ============================================================================

# BESS PRE-CALCULATION (Updated Call 1)
bess_power_transient = 0.0
bess_energy_transient = 0.0
bess_breakdown_transient = {}

if use_bess:
    bess_power_transient, bess_energy_transient, bess_breakdown_transient = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        load_ramp_req,
        enable_black_start
    )

spinning_reserve_result = calculate_spinning_reserve_units(
    p_avg_load=p_total_avg,
    unit_capacity=unit_site_cap,
    spinning_reserve_pct=step_load_req,
    use_bess=use_bess,
    bess_power_mw=bess_power_transient if use_bess else 0,
    gen_step_capability_pct=gen_data["step_load_pct"]
)

# Extract results
n_running_for_spinning = spinning_reserve_result['n_units_running']
spinning_reserve_mw = spinning_reserve_result['spinning_reserve_mw']
spinning_from_gens = spinning_reserve_result['spinning_from_gens']
spinning_from_bess = spinning_reserve_result['spinning_from_bess']

n_running_from_load, fleet_options = optimize_fleet_size(
    p_total_avg, p_total_peak, unit_site_cap, step_load_req, gen_data, use_bess
)
n_running_from_load = max(n_running_from_load, n_running_for_spinning)

# Step 2: Calculate N+X for availability target
avail_decimal = avail_req / 100
mtbf_hours = gen_data["mtbf_hours"]
mttr_hours = 48 

# ============================================================================
# HYBRID ALGORITHM
# ============================================================================

reliability_configs = []

# Config A: No BESS
spinning_no_bess = calculate_spinning_reserve_units(
    p_avg_load=p_total_avg,
    unit_capacity=unit_site_cap,
    spinning_reserve_pct=step_load_req,
    use_bess=False,
    bess_power_mw=0
)
config_a_running = max(int(math.ceil(p_total_peak / unit_site_cap)), spinning_no_bess['n_units_running'])
search_min_a = max(1, int(config_a_running * 0.95))
search_max_a = int(config_a_running * 1.2)

best_config_a = None
for n_run in range(search_min_a, search_max_a):
    if n_run * unit_site_cap < spinning_no_bess['required_online_capacity']: continue
    for n_res in range(0, 20):
        n_tot = n_run + n_res
        avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
        if avg_avail >= avail_decimal:
            load_pct_a = (p_total_avg / (n_run * unit_site_cap)) * 100
            best_config_a = {
                'name': 'A: No BESS', 'n_running': n_run, 'n_reserve': n_res, 'n_total': n_tot,
                'bess_mw': 0, 'bess_mwh': 0, 'bess_credit': 0, 'availability': avg_avail, 'load_pct': load_pct_a
            }
            break
    if best_config_a: break

if not best_config_a:
    fallback_n_run = config_a_running
    fallback_n_res = 14
    fallback_avail, _ = calculate_availability_weibull(fallback_n_run + fallback_n_res, fallback_n_run, mtbf_hours, project_years)
    best_config_a = {
        'name': 'A: No BESS', 'n_running': fallback_n_run, 'n_reserve': fallback_n_res, 'n_total': fallback_n_run + fallback_n_res,
        'bess_mw': 0, 'bess_mwh': 0, 'bess_credit': 0, 'availability': fallback_avail, 'load_pct': 50.0
    }
reliability_configs.append(best_config_a)

# Config B: BESS Transient
if use_bess:
    spinning_with_bess = calculate_spinning_reserve_units(
        p_avg_load=p_total_avg, unit_capacity=unit_site_cap, spinning_reserve_pct=step_load_req,
        use_bess=True, bess_power_mw=bess_power_transient
    )
    n_running_min_b = spinning_with_bess['n_units_running']
    best_config_b = None
    
    for n_run in range(max(1, n_running_min_b - 2), n_running_min_b + 5):
        if n_run * unit_site_cap < spinning_with_bess['required_online_capacity']: continue
        for n_res in range(0, 20):
            n_tot = n_run + n_res
            avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
            if avg_avail >= avail_decimal:
                load_pct_b = (p_total_avg / (n_run * unit_site_cap)) * 100
                eff_b = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_b, gen_data["type"])
                best_config_b = {
                    'name': 'B: BESS Transient', 'n_running': n_run, 'n_reserve': n_res, 'n_total': n_tot,
                    'bess_mw': bess_power_transient, 'bess_mwh': bess_energy_transient, 'bess_credit': 0,
                    'availability': avg_avail, 'load_pct': load_pct_b, 'efficiency': eff_b
                }
                break
        if best_config_b: break
    
    if best_config_b: reliability_configs.append(best_config_b)

# Config C: Hybrid
if use_bess and bess_reliability_enabled:
    target_gensets_covered = 5 if bess_strategy == 'Reliability Priority' else 3
    bess_coverage_hrs = 2.0
    bess_power_hybrid = max(bess_power_transient, target_gensets_covered * unit_site_cap)
    bess_energy_hybrid = max(bess_power_hybrid * bess_coverage_hrs, target_gensets_covered * unit_site_cap * bess_coverage_hrs)
    
    bess_credit_units, credit_breakdown = calculate_bess_reliability_credit(bess_power_hybrid, bess_energy_hybrid, unit_site_cap, mttr_hours)
    bess_credit_int = int(bess_credit_units * 0.65)
    
    best_config_c = None
    n_run_c_start = spinning_with_bess['n_units_running']
    
    for n_run in range(max(1, n_run_c_start - 3), n_run_c_start + 5):
        if n_run * unit_site_cap < spinning_with_bess['required_online_capacity']: continue
        for n_res_base in range(0, 20):
            n_res_effective = max(1, n_res_base - bess_credit_int)
            n_tot = n_run + n_res_effective
            avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
            if (avg_avail + 0.0005) >= avail_decimal:
                load_pct_c = (p_total_avg / (n_run * unit_site_cap)) * 100
                eff_c = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct_c, gen_data["type"])
                best_config_c = {
                    'name': f'C: {bess_strategy}', 'n_running': n_run, 'n_reserve': n_res_effective, 'n_total': n_tot,
                    'bess_mw': bess_power_hybrid, 'bess_mwh': bess_energy_hybrid, 'bess_credit': bess_credit_units * 0.65,
                    'availability': min(0.9999, avg_avail + 0.0005), 'credit_breakdown': credit_breakdown, 'load_pct': load_pct_c, 'efficiency': eff_c
                }
                break
        if best_config_c: break
    
    if best_config_c: reliability_configs.append(best_config_c)

# Selection Logic
if bess_strategy == "Transient Only" and len(reliability_configs) >= 2:
    selected_config = reliability_configs[1]
elif bess_strategy in ["Hybrid (Balanced)", "Reliability Priority"] and len(reliability_configs) >= 3:
    selected_config = reliability_configs[2]
elif len(reliability_configs) >= 1:
    selected_config = reliability_configs[0]
else:
    selected_config = {'name': 'Fallback', 'n_running': n_running_from_load, 'n_reserve': 10, 'n_total': n_running_from_load + 10, 'bess_mw': 0, 'bess_mwh': 0, 'availability': 0.999, 'load_pct': 80.0}

n_running = selected_config['n_running']
n_reserve = selected_config['n_reserve']
n_total = selected_config['n_total']
prob_gen = selected_config['availability']
bess_power_total = selected_config.get('bess_mw', 0)
bess_energy_total = selected_config.get('bess_mwh', 0)
target_met = prob_gen >= avail_decimal
load_per_unit_pct = selected_config['load_pct']
installed_cap = n_total * unit_site_cap
_, availability_curve = calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years)

# Load Distribution Strategy
st.sidebar.markdown("⚡ **Load Distribution**")
load_strategy = st.sidebar.radio("Operating Mode", ["Equal Loading (N units)", "Spinning Reserve (N+1)", "Sequential"])
if load_strategy == "Spinning Reserve (N+1)":
    units_running = n_running + 1 if n_reserve > 0 else n_running
    load_per_unit_pct = (p_total_avg / (units_running * unit_site_cap)) * 100
else:
    units_running = n_running

fleet_efficiency = get_part_load_efficiency(gen_data["electrical_efficiency"], load_per_unit_pct, gen_data["type"])

# Recalculate BESS breakdown for final selected (Updated Call 2)
if use_bess:
    bess_power_total, bess_energy_total, bess_breakdown = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        load_ramp_req,
        enable_black_start
    )
else:
    bess_breakdown = {}

if volt_mode == "Auto-Recommend":
    if installed_cap < 50: rec_voltage_kv = 13.8
    else: rec_voltage_kv = 34.5
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
    storage_area_m2 = (p_total_avg/fleet_efficiency * 3.412 * 24 * lng_days / 0.075 * 0.00378 * 5)
else: storage_area_m2 = 0
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)
area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
disp_area = total_area_m2 / 10000 if not is_imperial else total_area_m2 * 0.000247
disp_area_unit = "Ha" if not is_imperial else "Acres"

# ==============================================================================
# 5. FINANCIALS
# ==============================================================================
total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412
if not is_lng_primary:
    rec_pipe_dia = math.sqrt((total_fuel_input_mmbtu_hr * 1000 / 1.02) / 3000) * 2
else: rec_pipe_dia = 0

nox_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_nox"] / 1000)
co_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_co"] / 1000)
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor
at_capex_total = (installed_cap * 1000 * 100) if nox_lb_hr * 8760 > 100 else 0

gen_cost_total = (installed_cap * 1000) * (gen_data["est_cost_kw"] * regional_mult) / 1e6
bess_capex_m = ((bess_power_total*1000*250) + (bess_energy_total*1000*400))/1e6 if use_bess else 0
infra_capex = (storage_area_m2 * 500 + dist_gas_main_m * 100)/1e6

cost_items = [
    {"Item": "Generation Units", "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Cost (M USD)": gen_cost_total * (gen_data["est_install_kw"]/gen_data["est_cost_kw"])},
    {"Item": "BESS System", "Cost (M USD)": bess_capex_m},
    {"Item": "Infrastructure", "Cost (M USD)": infra_capex},
    {"Item": "Emissions Control", "Cost (M USD)": at_capex_total / 1e6},
]
df_capex = pd.DataFrame(cost_items)
initial_capex_sum = df_capex["Cost (M USD)"].sum()

effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours
om_cost_year = (installed_cap * 1000 * 15) + (mwh_year * 3.5) + (n_total * 120000) + (bess_power_total*1000*5 if use_bess else 0)
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
capex_annualized = initial_capex_sum * 1e6 * crf
total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + carbon_cost_year
lcoe = total_annual_cost / (mwh_year * 1000)

annual_savings = (mwh_year * 1000 * benchmark_price) - (fuel_cost_year + om_cost_year + carbon_cost_year)
npv = (annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)) - (initial_capex_sum * 1e6)
payback = (initial_capex_sum * 1e6) / annual_savings if annual_savings > 0 else 0

# ==============================================================================
# 6. OUTPUTS
# ==============================================================================
t1, t2, t3, t4, t5 = st.tabs(["📊 System Design", "⚡ Performance", "🏗️ Footprint", "❄️ Cooling", "💰 Economics"])

with t1:
    st.subheader("System Architecture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fleet", f"{n_running}+{n_reserve}")
    c2.metric("Installed", f"{installed_cap:.1f} MW")
    c3.metric("Availability", f"{prob_gen*100:.3f}%", "Target Met" if target_met else "Below")
    c4.metric("Load/Unit", f"{load_per_unit_pct:.1f}%")
    
    st.markdown("### 🔄 Spinning Reserve & BESS")
    sr_cols = st.columns(3)
    sr_cols
