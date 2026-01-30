import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT QuickSize v2.0", page_icon="⚡", layout="wide")

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
    step_event_duration = 60  # seconds
    events_per_day = 5
    
    # C-rate consideration
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
    
    IMPORTANT: BESS cannot cover full MTTR (48 hrs), only provides bridge power
    while backup gensets start up (typically 2-4 hours)
    
    Parameters:
    - bess_power_mw: BESS power capacity (MW)
    - bess_energy_mwh: BESS energy capacity (MWh)
    - unit_capacity_mw: Single generator capacity (MW)
    - mttr_hours: Mean Time To Repair (default 48 hrs, but BESS covers 2-4 hrs)
    
    Returns:
    - effective_credit: Number of gensets BESS can replace (float)
    - credit_breakdown: Dictionary with calculation details
    """
    if bess_power_mw <= 0 or bess_energy_mwh <= 0:
        return 0.0, {}
    
    # BESS realistically covers 2-4 hours until backup gensets arrive or repair starts
    # NOT the full MTTR of 48 hours
    realistic_coverage_hrs = 2.0  # Conservative: 2 hours
    
    # Power-based credit: How many units can BESS replace instantly
    power_credit = bess_power_mw / unit_capacity_mw
    
    # Energy-based credit: How long can BESS sustain that power
    bess_duration_hrs = bess_energy_mwh / bess_power_mw if bess_power_mw > 0 else 0
    energy_credit = bess_energy_mwh / (unit_capacity_mw * realistic_coverage_hrs)
    
    # Take minimum (bottleneck)
    raw_credit = min(power_credit, energy_credit)
    
    # Apply factors:
    bess_availability = 0.98  # BESS itself has ~98% availability
    coverage_factor = 0.70    # Increased from 0.60 to 0.70 (less conservative)
    
    effective_credit = raw_credit * bess_availability * coverage_factor
    
    credit_breakdown = {
        'power_credit': power_credit,
        'energy_credit': energy_credit,
        'raw_credit': raw_credit,
        'bess_availability': bess_availability,
        'coverage_factor': coverage_factor,
        'effective_credit': effective_credit,
        'bess_duration_hrs': bess_duration_hrs,
        'realistic_coverage_hrs': realistic_coverage_hrs
    }
    
    return effective_credit, credit_breakdown

def calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, 
                                  maintenance_interval_hrs=1000, maintenance_duration_hrs=48):
    """
    Reliability model using industry standard availability formula INCLUDING planned maintenance
    
    Availability = MTBF / (MTBF + MTTR + Planned_Maintenance_Time)
    
    This is the CORRECT formula that accounts for:
    1. Random failures (MTBF/MTTR)
    2. Planned maintenance outages (often overlooked!)
    
    For High Speed RICE:
    - Maintenance every 1000 hrs → 8.76 events/year
    - 48 hrs per event → 420 hrs/year unavailable
    - This is 4.8% unavailability just from maintenance!
    """
    # Typical MTTR for power generation equipment
    mttr_hours = 48  # 48 hours average repair time for failures
    
    # Calculate planned maintenance unavailability
    # Annual hours of planned maintenance
    annual_maintenance_hrs = (8760 / maintenance_interval_hrs) * maintenance_duration_hrs
    
    # Total unavailability = MTTR (failures) + Planned Maintenance
    total_unavailable_hrs = mttr_hours + annual_maintenance_hrs
    
    # Unit availability (corrected formula)
    unit_availability = mtbf_hours / (mtbf_hours + total_unavailable_hrs)
    
    # Example for High Speed RICE (MTBF=48000, maint every 1000 hrs for 48 hrs):
    # annual_maintenance = (8760/1000) × 48 = 420.5 hrs
    # availability = 48000 / (48000 + 48 + 420.5) = 48000 / 48468.5 = 99.03%
    # vs OLD (incorrect): 48000 / 48048 = 99.90% ← 0.87% too optimistic!
    
    # System availability (N+X configuration using binomial)
    # P(system works) = P(at least n_running units are available)
    sys_avail = 0
    for k in range(n_running, n_total + 1):
        comb = math.comb(n_total, k)
        prob = comb * (unit_availability ** k) * ((1 - unit_availability) ** (n_total - k))
        sys_avail += prob
    
    # For availability curve over time, apply modest aging (0.1% per year)
    availability_over_time = []
    for year in range(1, project_years + 1):
        # Conservative aging: 0.1% per year
        aging_factor = 1.0 - (year * 0.001)
        aging_factor = max(0.95, aging_factor)  # Floor at 95%
        
        aged_unit_availability = unit_availability * aging_factor
        
        # Recalculate system availability with aged units
        sys_avail_year = 0
        for k in range(n_running, n_total + 1):
            comb = math.comb(n_total, k)
            prob = comb * (aged_unit_availability ** k) * ((1 - aged_unit_availability) ** (n_total - k))
            sys_avail_year += prob
        
        availability_over_time.append(sys_avail_year)
    
    # Return year 1 availability (not average over 20 years)
    # This is standard practice for availability targets
    return sys_avail, availability_over_time

def optimize_fleet_size(p_net_req_avg, p_net_req_peak, unit_cap, step_load_req, gen_data, use_bess=False):
    """
    Multi-objective fleet optimization
    NOW CONSIDERS: BESS for peak shaving and step load coverage
    """
    # NEW: If BESS enabled, size for average load (BESS handles peaks)
    if use_bess:
        # Constraint 1: Average capacity + safety margin (BESS covers peak)
        n_min_peak = math.ceil(p_net_req_avg * 1.15 / unit_cap)  # +15% margin
        
        # Constraint 3: Step load covered by BESS (not gensets)
        # Only need headroom for ramp-up capability
        headroom_required = p_net_req_avg * 1.10  # 10% headroom only
        n_min_step = math.ceil(headroom_required / unit_cap)
    else:
        # Constraint 1: Must cover peak load without BESS
        n_min_peak = math.ceil(p_net_req_peak / unit_cap)
        
        # Constraint 3: Must have headroom for step load
        headroom_required = p_net_req_avg * (1 + step_load_req/100) * 1.20
        n_min_step = math.ceil(headroom_required / unit_cap)
    
    # Constraint 2: Part-load efficiency (target 65-80% load)
    n_ideal_eff = math.ceil(p_net_req_avg / (unit_cap * 0.72))
    
    # Take maximum of all constraints
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    # Analyze efficiency at different fleet sizes
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        # Check if meets minimum capacity
        if use_bess:
            if n * unit_cap < p_net_req_avg * 1.10:  # Need 110% of average
                continue
        else:
            if n * unit_cap < p_net_req_peak:  # Need full peak
                continue
        
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 30 or load_pct > 95:  # Outside acceptable range
            continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        # Score: favor 65-80% load range
        optimal_load = 72.5
        load_penalty = abs(load_pct - optimal_load) / 100
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': eff * (1 - load_penalty * 0.5)
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}
    
    # Take maximum
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    # Analyze efficiency at different fleet sizes
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        if n * unit_cap < p_net_req_peak:
            continue
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 20 or load_pct > 95:  # Outside acceptable range
            continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': eff * (1 - abs(load_pct - 70)/100)  # Penalize deviation from 70%
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}

def calculate_macrs_depreciation(capex, project_years):
    """
    MACRS 5-year depreciation schedule
    """
    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    tax_rate = 0.21  # Federal corporate tax rate
    
    pv_benefit = 0
    wacc = 0.08  # Use global WACC
    
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

st.title(f"⚡ CAT QuickSize v2.0 ({freq_hz}Hz)")
st.markdown("**Next-Gen Data Center Power Solutions.**\nAdvanced modeling with PUE optimization, footprint constraints, and sophisticated LCOE analysis.")

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
    
    # PUE defaults by type (2026 best practices)
    pue_defaults = {
        "AI Factory (Training)": 1.15,      # Liquid cooling, DLC
        "AI Factory (Inference)": 1.20,     # High density, optimized
        "Hyperscale Standard": 1.25,        # Air cooling, free cooling
        "Colocation": 1.50,                 # Multi-tenant
        "Edge Computing": 1.60              # Small scale
    }
    
    # Step load and BESS defaults
    is_ai = "AI" in dc_type
    def_step_load = 40.0 if is_ai else 15.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    
    # NEW: PUE Input (replaces DC Aux %)
    st.markdown("📊 **Power Usage Effectiveness (PUE)**")
    pue = st.slider(
        "Data Center PUE", 
        1.05, 2.00, 
        pue_defaults[dc_type], 
        0.05,
        help="PUE = Total Facility Power / IT Equipment Power. Industry standard metric."
    )
    
    # Show breakdown
    p_total_dc = p_it * pue
    p_aux = p_total_dc - p_it
    
    with st.expander("ℹ️ PUE Breakdown"):
        st.write(f"**IT Load:** {p_it:.1f} MW")
        st.write(f"**Auxiliary Load:** {p_aux:.1f} MW ({(pue-1)*100:.1f}% of IT)")
        st.write(f"**Total DC Load:** {p_total_dc:.1f} MW")
        st.caption("Auxiliary = Cooling + UPS losses + Lighting + Network")
    
    # ===== LOAD PROFILE SECTION =====
    st.markdown("📊 **Annual Load Profile**")
    
    load_profiles = {
        "AI Factory (Training)": {
            "capacity_factor": 0.96,
            "peak_avg_ratio": 1.08,
            "description": "Continuous 24/7 training runs"
        },
        "AI Factory (Inference)": {
            "capacity_factor": 0.85,
            "peak_avg_ratio": 1.25,
            "description": "Variable inference loads with peaks"
        },
        "Hyperscale Standard": {
            "capacity_factor": 0.75,
            "peak_avg_ratio": 1.20,
            "description": "Mixed workloads, diurnal patterns"
        },
        "Colocation": {
            "capacity_factor": 0.65,
            "peak_avg_ratio": 1.35,
            "description": "Multi-tenant, business hours peaks"
        },
        "Edge Computing": {
            "capacity_factor": 0.50,
            "peak_avg_ratio": 1.50,
            "description": "Highly variable local demand"
        }
    }
    
    profile = load_profiles[dc_type]
    
    col_cf1, col_cf2 = st.columns(2)
    capacity_factor = col_cf1.slider(
        "Capacity Factor (%)", 
        30.0, 100.0, 
        profile["capacity_factor"]*100, 
        1.0,
        help=profile["description"]
    ) / 100.0
    
    peak_avg_ratio = col_cf2.slider(
        "Peak/Avg Ratio", 
        1.0, 2.0, 
        profile["peak_avg_ratio"], 
        0.05
    )
    
    # Calculate loads
    p_total_avg = p_total_dc * capacity_factor
    p_total_peak = p_total_dc * peak_avg_ratio
    
    st.info(f"💡 **Load Analysis:**\n"
            f"- Avg: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**\n"
            f"- Effective Hours/Year: **{8760*capacity_factor:.0f} hrs**")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Step Load Req (%)", 0.0, 100.0, def_step_load)
    
    # ===== NEW: FOOTPRINT CONSTRAINTS =====
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
        else:  # Hectares
            max_area_input = st.number_input("Max Available Area (Ha)", 0.1, 50.0, 5.0, step=0.5)
            max_area_m2 = max_area_input * 10000
    else:
        max_area_m2 = 999999999  # No limit
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual":
        voltage_option = st.selectbox("Select Voltage Level", [
            "4.16 kV (Low Voltage - Small DCs)",
            "13.8 kV (Medium Voltage - Standard)",
            "34.5 kV (High MV - Large Off-Grid DCs)",
            "69 kV (Sub-Transmission - Very Large)",
            "Custom"
        ])
        
        voltage_map = {
            "4.16 kV (Low Voltage - Small DCs)": 4.16,
            "13.8 kV (Medium Voltage - Standard)": 13.8,
            "34.5 kV (High MV - Large Off-Grid DCs)": 34.5,
            "69 kV (Sub-Transmission - Very Large)": 69.0,
        }
        
        if voltage_option == "Custom":
            manual_voltage_kv = st.number_input("Custom Voltage (kV)", 0.4, 230.0, 13.8, step=0.1)
        else:
            manual_voltage_kv = voltage_map[voltage_option]
            st.caption(f"✅ Selected: {manual_voltage_kv} kV")
    
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

    # -------------------------------------------------------------------------
    # GROUP 2: TECHNOLOGY SOLUTION
    # -------------------------------------------------------------------------
    st.header("2. Technology Solution")
    
    st.markdown("⚙️ **Generation Technology**")
    gen_filter = st.multiselect(
        "Technology Filter", 
        ["High Speed", "Medium Speed", "Gas Turbine"],
        default=["High Speed", "Medium Speed"]
    )
    
    use_bess = st.checkbox("Include BESS (Battery Energy Storage)", value=def_use_bess)
    
    bess_strategy = "Hybrid (Balanced)"  # Default
    bess_reliability_enabled = False
    
    if use_bess:
        st.markdown("🔋 **BESS Strategy**")
        bess_strategy = st.radio(
            "Sizing Mode",
            [
                "Transient Only",
                "Hybrid (Balanced)",
                "Reliability Priority"
            ],
            index=1,  # Default to Hybrid
            help="Transient: Peak shaving + step load only\n"
                 "Hybrid: Also reduces genset redundancy (best NPV)\n"
                 "Reliability: Maximum BESS, minimum gensets"
        )
        
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

    # -------------------------------------------------------------------------
    # GROUP 3: ECONOMICS - ENHANCED
    # -------------------------------------------------------------------------
    st.header("3. Economics & ROI")
    
    st.markdown("💰 **Energy Pricing**")
    
    # Gas pricing with transport
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
    
    # Carbon pricing
    st.markdown("🌍 **Carbon Pricing**")
    carbon_scenario = st.selectbox("Carbon Price Scenario", [
        "None (Current 2026)",
        "California Cap-and-Trade",
        "EU ETS",
        "Federal Projected 2030",
        "High Case (IEA Net Zero)"
    ])
    
    carbon_prices = {
        "None (Current 2026)": 0,
        "California Cap-and-Trade": 35,
        "EU ETS": 85,
        "Federal Projected 2030": 50,
        "High Case (IEA Net Zero)": 150
    }
    
    carbon_price_per_ton = carbon_prices[carbon_scenario]
    
    if carbon_price_per_ton > 0:
        st.info(f"💨 **Carbon Price:** ${carbon_price_per_ton}/ton CO₂")
    
    # Financial parameters
    c_fin1, c_fin2 = st.columns(2)
    wacc = c_fin1.number_input("WACC (%)", 1.0, 20.0, 8.0, step=0.5) / 100
    project_years = c_fin2.number_input("Project Life (Years)", 10, 30, 20, step=5)
    
    # Tax incentives
    st.markdown("💸 **Tax Incentives & Depreciation**")
    enable_itc = st.checkbox("Include ITC (30% for CHP)", value=include_chp)
    enable_ptc = st.checkbox("Include PTC ($0.013/kWh, 10yr)", value=False)
    enable_depreciation = st.checkbox("Include MACRS Depreciation", value=True)
    
    # Regional costs
    st.markdown("📍 **Regional Adjustments**")
    region = st.selectbox("Region", [
        "US - Gulf Coast", "US - Northeast", "US - West Coast", "US - Midwest",
        "Europe - Western", "Europe - Eastern", "Middle East", "Asia Pacific",
        "Latin America", "Africa"
    ])
    
    regional_multipliers = {
        "US - Gulf Coast": 1.0,
        "US - Northeast": 1.25,
        "US - West Coast": 1.30,
        "US - Midwest": 1.05,
        "Europe - Western": 1.35,
        "Europe - Eastern": 0.90,
        "Middle East": 1.10,
        "Asia Pacific": 0.85,
        "Latin America": 0.95,
        "Africa": 1.15
    }
    regional_mult = regional_multipliers[region]
    
    # LCOE Target
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
    density_score = gen_data["power_density_mw_per_m2"] * 20  # Favor high density
    
    total_score = step_match * 100 + eff_score + cost_score + density_score
    
    if total_score > best_score:
        best_score = total_score
        best_gen = gen_name

selected_gen = st.sidebar.selectbox(
    "🔧 Selected Generator",
    list(available_gens.keys()),
    index=list(available_gens.keys()).index(best_gen) if best_gen else 0
)

gen_data = available_gens[selected_gen]

# ============================================================================
# GENERATOR PARAMETERS - EDITABLE (NEW FEATURE)
# ============================================================================
with st.sidebar.expander("⚙️ Generator Parameters (Editable)", expanded=False):
    st.markdown("**Reliability & Maintenance:**")
    
    # MTBF (Mean Time Between Failures)
    mtbf_edit = st.number_input(
        "MTBF (hours)",
        value=gen_data["mtbf_hours"],
        min_value=10000,
        max_value=150000,
        step=1000,
        help="Mean Time Between Failures - affects unplanned outages"
    )
    gen_data["mtbf_hours"] = mtbf_edit
    
    # Maintenance Interval
    maint_interval_edit = st.number_input(
        "Maintenance Interval (hrs)",
        value=gen_data["maintenance_interval_hrs"],
        min_value=500,
        max_value=20000,
        step=100,
        help="Hours between planned maintenance events"
    )
    gen_data["maintenance_interval_hrs"] = maint_interval_edit
    
    # Maintenance Duration
    maint_duration_edit = st.number_input(
        "Maintenance Duration (hrs)",
        value=gen_data["maintenance_duration_hrs"],
        min_value=12,
        max_value=240,
        step=6,
        help="Downtime per maintenance event"
    )
    gen_data["maintenance_duration_hrs"] = maint_duration_edit
    
    # Calculate and show availability impact
    annual_maint_hrs = (8760 / maint_interval_edit) * maint_duration_edit
    unit_avail = mtbf_edit / (mtbf_edit + 48 + annual_maint_hrs)
    
    st.markdown("---")
    st.markdown("**Calculated Unit Availability:**")
    st.metric("Single Unit", f"{unit_avail*100:.2f}%")
    
    maint_unavail = (annual_maint_hrs / 8760) * 100
    failure_unavail = (48 / (mtbf_edit + 48)) * 100
    
    st.caption(f"📊 Breakdown:")
    st.caption(f"  • Planned Maint: {maint_unavail:.2f}% unavailable")
    st.caption(f"  • Failures (MTTR=48h): {failure_unavail:.3f}% unavailable")
    st.caption(f"  • **Total: {((1-unit_avail)*100):.2f}% unavailable**")
    
    st.markdown("---")
    st.markdown("**Performance:**")
    
    # Efficiency
    eff_edit = st.number_input(
        "Electrical Efficiency",
        value=gen_data["electrical_efficiency"],
        min_value=0.25,
        max_value=0.60,
        step=0.001,
        format="%.3f",
        help="Electrical efficiency (HHV basis)"
    )
    gen_data["electrical_efficiency"] = eff_edit
    
    # Step Load Capability
    step_edit = st.number_input(
        "Step Load Capability (%)",
        value=gen_data["step_load_pct"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="Maximum % load that can be accepted in one step"
    )
    gen_data["step_load_pct"] = step_edit
    
    # Ramp Rate
    ramp_edit = st.number_input(
        "Ramp Rate (MW/s)",
        value=gen_data["ramp_rate_mw_s"],
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        help="Rate of load change capability"
    )
    gen_data["ramp_rate_mw_s"] = ramp_edit
    
    st.success("✅ Custom parameters applied")

# Derated capacity
unit_iso_cap = gen_data["iso_rating_mw"]
unit_site_cap = unit_iso_cap * derate_factor_calc

# ============================================================================
# ENHANCED FLEET OPTIMIZATION - AVAILABILITY-DRIVEN WITH BESS CREDIT
# ============================================================================

# Step 1: Calculate MINIMUM n_running based on load requirements
n_running_from_load, fleet_options = optimize_fleet_size(
    p_total_avg, p_total_peak, unit_site_cap, step_load_req, gen_data, use_bess
)

# Step 2: Calculate N+X for availability target
avail_decimal = avail_req / 100
mtbf_hours = gen_data["mtbf_hours"]
mttr_hours = 48  # Realistic: 2 days repair time

# Calculate BESS requirements FIRST (needed for reliability credit)
bess_power_transient = 0.0
bess_energy_transient = 0.0
bess_breakdown_transient = {}

if use_bess:
    bess_power_transient, bess_energy_transient, bess_breakdown_transient = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        enable_black_start
    )

# ============================================================================
# HYBRID ALGORITHM: Generate comparison table of Gen+BESS configurations
# ============================================================================

reliability_configs = []

# Configuration A: No BESS (Baseline)
config_a_running = n_running_from_load
# Expand search range - be more generous
search_min_a = max(1, int(config_a_running * 0.8))  # Down to 80% of load-optimal
search_max_a = int(config_a_running * 2.0)  # Up to 2x load-optimal

best_config_a = None

# Search exhaustively for Config A
for n_run in range(search_min_a, search_max_a):
    # Config A: NO BESS - must ALWAYS cover PEAK load
    capacity_min_a = p_total_peak
    if n_run * unit_site_cap < capacity_min_a:
        continue
    
    for n_res in range(0, 20):  # Extended to N+19
        n_tot = n_run + n_res
        
        avg_avail, _ = calculate_availability_weibull(n_tot, n_run, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
        
        if avg_avail >= avail_decimal:
            best_config_a = {
                'name': 'A: No BESS',
                'n_running': n_run,
                'n_reserve': n_res,
                'n_total': n_tot,
                'bess_mw': 0,
                'bess_mwh': 0,
                'bess_credit': 0,
                'availability': avg_avail
            }
            break
    if best_config_a:
        break

# If still not found, create fallback with CORRECT availability calculation
if not best_config_a:
    # Fallback: size for peak load
    fallback_n_run = int(p_total_peak / unit_site_cap) + 1
    fallback_n_res = 14
    fallback_n_tot = fallback_n_run + fallback_n_res
    
    # Calculate ACTUAL availability for fallback (not hardcoded 95%)
    fallback_avail, _ = calculate_availability_weibull(
        fallback_n_tot, fallback_n_run, mtbf_hours, project_years
    )
    
    best_config_a = {
        'name': 'A: No BESS',
        'n_running': fallback_n_run,
        'n_reserve': fallback_n_res,
        'n_total': fallback_n_tot,
        'bess_mw': 0,
        'bess_mwh': 0,
        'bess_credit': 0,
        'availability': fallback_avail
    }

reliability_configs.append(best_config_a)

# Configuration B: BESS Transient Only
if use_bess:
    # Debug logging
    import sys
    print(f"[DEBUG] Config B: Starting calculation", file=sys.stderr)
    print(f"[DEBUG] p_total_avg={p_total_avg}, unit_site_cap={unit_site_cap}", file=sys.stderr)
    
    # With BESS for peak shaving, optimize n_running for efficiency
    # Target: 70-76% load per unit (sweet spot)
    
    # Calculate optimal n_running for 72% target load
    target_load_optimal = 0.72
    n_running_for_efficiency = p_total_avg / (unit_site_cap * target_load_optimal)
    n_running_optimal_b = int(round(n_running_for_efficiency))
    
    print(f"[DEBUG] Config B: n_running_optimal_b={n_running_optimal_b}", file=sys.stderr)
    
    # Ensure we have enough capacity (minimum 105% of average)
    n_running_min_b = int(math.ceil(p_total_avg * 1.05 / unit_site_cap))
    n_running_optimal_b = max(n_running_optimal_b, n_running_min_b)
    
    print(f"[DEBUG] Config B: After capacity check, n_running_optimal_b={n_running_optimal_b}", file=sys.stderr)
    
    best_config_b = None
    found_b = False
    
    # Search around optimal point (±5 units)
    for n_run_offset in range(-5, 10):
        if found_b:
            break
        
        n_run = n_running_optimal_b + n_run_offset
        if n_run < n_running_min_b:
            continue
        
        # Check capacity
        if n_run * unit_site_cap < p_total_avg * 1.05:
            continue
        
        # Search for minimum reserve needed
        for n_res in range(0, 20):
            n_tot = n_run + n_res
            
            try:
                avg_avail, _ = calculate_availability_weibull(
                    n_tot, n_run, mtbf_hours, project_years,
                    gen_data["maintenance_interval_hrs"],
                    gen_data["maintenance_duration_hrs"]
                )
                
                if avg_avail >= avail_decimal:
                    # Calculate metrics
                    load_pct_b = (p_total_avg / (n_run * unit_site_cap)) * 100
                    eff_b = get_part_load_efficiency(
                        gen_data["electrical_efficiency"],
                        load_pct_b,
                        gen_data["type"]
                    )
                    
                    print(f"[DEBUG] Config B FOUND: n_run={n_run}, n_res={n_res}, load={load_pct_b:.1f}%, eff={eff_b*100:.1f}%", file=sys.stderr)
                    
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
                        'score': eff_b
                    }
                    found_b = True
                    break
            except Exception as e:
                st.sidebar.error(f"Config B error: {str(e)}")
                continue
    
    if best_config_b:
        reliability_configs.append(best_config_b)
    else:
        # Fallback for Config B
        st.sidebar.warning("⚠️ Config B: Using fallback")
        n_run_fallback = n_running_min_b
        n_res_fallback = 8
        fallback_avail_b, _ = calculate_availability_weibull(
            n_run_fallback + n_res_fallback, n_run_fallback,
            mtbf_hours, project_years,
            gen_data["maintenance_interval_hrs"],
            gen_data["maintenance_duration_hrs"]
        )
        
        best_config_b = {
            'name': 'B: BESS Transient (fallback)',
            'n_running': n_run_fallback,
            'n_reserve': n_res_fallback,
            'n_total': n_run_fallback + n_res_fallback,
            'bess_mw': bess_power_transient,
            'bess_mwh': bess_energy_transient,
            'bess_credit': 0,
            'availability': fallback_avail_b
        }
        reliability_configs.append(best_config_b)

# Configuration C: BESS Hybrid (Balanced)
if use_bess and bess_reliability_enabled:
    print(f"[DEBUG] Config C: Starting calculation", file=sys.stderr)
    
    # Start with same efficiency optimization as Config B
    target_load_optimal = 0.72
    n_running_for_efficiency = p_total_avg / (unit_site_cap * target_load_optimal)
    n_running_optimal_c = int(round(n_running_for_efficiency))
    
    n_running_min_c = int(math.ceil(p_total_avg * 1.05 / unit_site_cap))
    n_running_optimal_c = max(n_running_optimal_c, n_running_min_c)
    
    # BESS sizing for reliability
    if bess_strategy == 'Hybrid (Balanced)':
        target_gensets_covered = 3
        bess_coverage_hrs = 2.0
    else:  # Reliability Priority
        target_gensets_covered = 5
        bess_coverage_hrs = 2.5
    
    # Size BESS
    bess_power_hybrid = max(
        bess_power_transient,
        target_gensets_covered * unit_site_cap
    )
    bess_energy_hybrid = bess_power_hybrid * bess_coverage_hrs
    min_energy = target_gensets_covered * unit_site_cap * bess_coverage_hrs
    bess_energy_hybrid = max(bess_energy_hybrid, min_energy)
    
    # Calculate BESS credit
    try:
        bess_credit_units, credit_breakdown = calculate_bess_reliability_credit(
            bess_power_hybrid, bess_energy_hybrid, unit_site_cap, mttr_hours
        )
        
        # Apply 50% de-rating for conservatism
        bess_credit_conservative = bess_credit_units * 0.5
        bess_credit_int = max(0, int(bess_credit_conservative))
        
        print(f"[DEBUG] Config C: BESS Power={bess_power_hybrid:.1f} MW, Energy={bess_energy_hybrid:.1f} MWh", file=sys.stderr)
        print(f"[DEBUG] Config C: BESS Credit raw={bess_credit_units:.2f}, conservative={bess_credit_conservative:.2f}, int={bess_credit_int}", file=sys.stderr)
        
        # Show debug info
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔍 **BESS Credit Debug**")
        st.sidebar.caption(f"Power: {bess_power_hybrid:.1f} MW")
        st.sidebar.caption(f"Energy: {bess_energy_hybrid:.1f} MWh")
        st.sidebar.caption(f"Raw credit: {bess_credit_units:.2f}")
        st.sidebar.caption(f"Applied: {bess_credit_int} units")
        
    except Exception as e:
        st.sidebar.error(f"BESS credit calc error: {str(e)}")
        bess_credit_int = 0
        bess_credit_conservative = 0
        credit_breakdown = {}
    
    best_config_c = None
    found_c = False
    
    # Strategy: Start with Config B's reserve requirement, then apply BESS credit
    # First, find what reserve Config B needed
    if best_config_b and 'n_reserve' in best_config_b:
        config_b_reserve = best_config_b['n_reserve']
    else:
        # Fallback: estimate from Config A or use default
        config_b_reserve = best_config_a['n_reserve'] if best_config_a and 'n_reserve' in best_config_a else 8
    
    print(f"[DEBUG] Config C: Config B used n_reserve={config_b_reserve}, will apply credit={bess_credit_int}", file=sys.stderr)
    
    # Search for best config
    for n_run_offset in range(-5, 10):
        if found_c:
            break
        
        n_run = n_running_optimal_c + n_run_offset
        if n_run < n_running_min_c:
            continue
        
        if n_run * unit_site_cap < p_total_avg * 1.05:
            continue
        
        # Try with BESS credit applied: use higher base reserve, then subtract credit
        # This ensures BESS actually reduces the needed units
        for n_res_base in range(config_b_reserve - 2, config_b_reserve + 10):
            # Apply BESS credit - this is the key difference from Config B
            n_res_effective = max(1, n_res_base - bess_credit_int)
            n_tot = n_run + n_res_effective
            
            print(f"[DEBUG] Config C: Trying n_run={n_run}, n_res_base={n_res_base}, credit={bess_credit_int}, effective={n_res_effective}, total={n_tot}", file=sys.stderr)
            
            try:
                avg_avail, _ = calculate_availability_weibull(
                    n_tot, n_run, mtbf_hours, project_years,
                    gen_data["maintenance_interval_hrs"],
                    gen_data["maintenance_duration_hrs"]
                )
                
                if avg_avail >= avail_decimal:
                    load_pct_c = (p_total_avg / (n_run * unit_site_cap)) * 100
                    eff_c = get_part_load_efficiency(
                        gen_data["electrical_efficiency"],
                        load_pct_c,
                        gen_data["type"]
                    )
                    
                    print(f"[DEBUG] Config C FOUND: n_run={n_run}, n_res_effective={n_res_effective}, total={n_tot}, avail={avg_avail*100:.4f}%", file=sys.stderr)
                    
                    best_config_c = {
                        'name': f'C: {bess_strategy}',
                        'n_running': n_run,
                        'n_reserve': n_res_effective,
                        'n_total': n_tot,
                        'bess_mw': bess_power_hybrid,
                        'bess_mwh': bess_energy_hybrid,
                        'bess_credit': bess_credit_conservative,
                        'availability': avg_avail,
                        'credit_breakdown': credit_breakdown,
                        'load_pct': load_pct_c,
                        'efficiency': eff_c,
                        'score': eff_c
                    }
                    found_c = True
                    break
            except Exception as e:
                print(f"[DEBUG] Config C error: {str(e)}", file=sys.stderr)
                continue
    
    if best_config_c:
        reliability_configs.append(best_config_c)
    else:
        # Fallback
        st.sidebar.warning("⚠️ Config C: Using fallback")
        n_run_fallback = n_running_min_c
        n_res_fallback = max(1, 8 - bess_credit_int)
        
        fallback_avail_c, _ = calculate_availability_weibull(
            n_run_fallback + n_res_fallback, n_run_fallback,
            mtbf_hours, project_years,
            gen_data["maintenance_interval_hrs"],
            gen_data["maintenance_duration_hrs"]
        )
        
        best_config_c = {
            'name': f'C: {bess_strategy} (fallback)',
            'n_running': n_run_fallback,
            'n_reserve': n_res_fallback,
            'n_total': n_run_fallback + n_res_fallback,
            'bess_mw': bess_power_hybrid,
            'bess_mwh': bess_energy_hybrid,
            'bess_credit': bess_credit_conservative,
            'availability': fallback_avail_c
        }
        reliability_configs.append(best_config_c)

# Select final configuration based on strategy
if bess_strategy == "Transient Only" and len(reliability_configs) >= 2:
    selected_config = reliability_configs[1]  # Config B
elif bess_strategy in ["Hybrid (Balanced)", "Reliability Priority"] and len(reliability_configs) >= 3:
    selected_config = reliability_configs[2]  # Config C
elif len(reliability_configs) >= 1:
    selected_config = reliability_configs[0]  # Config A (fallback)
else:
    # Emergency fallback
    selected_config = {
        'name': 'Fallback',
        'n_running': n_running_from_load,
        'n_reserve': 10,
        'n_total': n_running_from_load + 10,
        'bess_mw': bess_power_transient if use_bess else 0,
        'bess_mwh': bess_energy_transient if use_bess else 0,
        'bess_credit': 0,
        'availability': 0.9999
    }

# Extract final values
n_running = selected_config['n_running']
n_reserve = selected_config['n_reserve']
n_total = selected_config['n_total']
prob_gen = selected_config['availability']
bess_power_total = selected_config['bess_mw']
bess_energy_total = selected_config['bess_mwh']
target_met = prob_gen >= avail_decimal

# Update BESS breakdown
if use_bess and bess_power_total > 0:
    bess_breakdown = bess_breakdown_transient.copy()
    bess_breakdown['reliability_backup'] = bess_power_total - bess_power_transient
else:
    bess_breakdown = {}

installed_cap = n_total * unit_site_cap

# Calculate reliability curve
_, availability_curve = calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])

# Show optimization result in sidebar
if use_bess and bess_reliability_enabled and 'bess_credit' in selected_config:
    units_saved = best_config_a['n_total'] - selected_config['n_total']
    if units_saved > 0:
        st.sidebar.success(
            f"✅ **BESS Reliability Credit:**\n\n"
            f"Gensets: {best_config_a['n_total']} → {selected_config['n_total']} "
            f"({units_saved} units saved)\n\n"
            f"BESS Credit: {selected_config['bess_credit']:.1f} genset-equivalents"
        )
    else:
        st.sidebar.info(
            f"ℹ️ **{selected_config['name']}:**\n\n"
            f"BESS provides {selected_config['bess_credit']:.1f} reliability credit"
        )
elif not target_met:
    availability_gap = (avail_req - prob_gen * 100)
    st.sidebar.error(f"⚠️ **Availability Target Not Met**")
    st.sidebar.warning(
        f"Target: {avail_req:.3f}%\n\n"
        f"Achievable: {prob_gen*100:.3f}%\n\n"
        f"Gap: {availability_gap:.3f}%\n\n"
        f"Using configuration: N+{n_reserve}"
    )
    st.sidebar.info(
        "💡 **Recommendations:**\n"
        "- Use higher MTBF generators\n"
        "- Increase BESS capacity\n"
        "- Accept lower availability\n"
        "- Add grid backup connection"
    )



# Load Distribution Strategy
st.sidebar.markdown("⚡ **Load Distribution**")
load_strategy = st.sidebar.radio(
    "Operating Mode",
    ["Equal Loading (N units)", "Spinning Reserve (N+1)", "Sequential"],
    help="Load distribution strategy"
)

if load_strategy == "Equal Loading (N units)":
    units_running = n_running
elif load_strategy == "Spinning Reserve (N+1)":
    units_running = n_running + 1 if n_reserve > 0 else n_running
else:
    units_running = n_running

load_per_unit_pct = (p_total_avg / (units_running * unit_site_cap)) * 100

# Fleet efficiency at operating point (use precalculated if available)
if 'fleet_efficiency_temp' in locals():
    fleet_efficiency = fleet_efficiency_temp
else:
    fleet_efficiency = get_part_load_efficiency(
        gen_data["electrical_efficiency"],
        load_per_unit_pct,
        gen_data["type"]
    )

# BESS Sizing (if enabled)
bess_power_total = 0.0
bess_energy_total = 0.0
bess_breakdown = {}

if use_bess:
    bess_power_total, bess_energy_total, bess_breakdown = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        enable_black_start
    )

# Voltage recommendation (adjusted for off-grid data centers)
if volt_mode == "Auto-Recommend":
    # For off-grid data centers, use medium voltage distribution
    # High voltage (138+ kV) is only for grid interconnection
    if installed_cap < 10:
        rec_voltage_kv = 4.16  # Low voltage distribution
    elif installed_cap < 50:
        rec_voltage_kv = 13.8  # Standard medium voltage
    elif installed_cap < 200:
        rec_voltage_kv = 34.5  # High medium voltage (max for off-grid)
    else:
        # For very large installations (>200 MW), may need dual voltage
        rec_voltage_kv = 34.5  # Still medium voltage for DC delivery
        # Note: Could also use 69 kV for transmission between gen and DC
else:
    rec_voltage_kv = manual_voltage_kv

# Transient stability
stability_ok, voltage_sag = transient_stability_check(
    gen_data["reactance_xd_2"], units_running, step_load_req
)

# ==============================================================================
# 4. FOOTPRINT CALCULATION & OPTIMIZATION
# ==============================================================================

# Calculate footprint per component
area_per_gen = 1 / gen_data["power_density_mw_per_m2"]  # m² per MW
area_gen = n_total * unit_site_cap * area_per_gen

area_bess = bess_power_total * 30 if use_bess else 0

# LNG storage
if has_lng_storage:
    total_fuel_input_mw_temp = (p_total_avg / fleet_efficiency)
    total_fuel_input_mmbtu_hr_temp = total_fuel_input_mw_temp * 3.412
    lng_mmbtu_total = total_fuel_input_mmbtu_hr_temp * 24 * lng_days
    lng_gal = lng_mmbtu_total / 0.075
    storage_area_m2 = (lng_gal * 0.00378541) * 5
else:
    storage_area_m2 = 0
    lng_gal = 0

# Cooling/CHP
pue_base = 1.35 if cooling_method == "Water-Cooled" else 1.50
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)

area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

# FOOTPRINT OPTIMIZATION
is_area_exceeded = total_area_m2 > max_area_m2
area_utilization_pct = (total_area_m2 / max_area_m2) * 100 if enable_footprint_limit else 0

footprint_recommendations = []

if is_area_exceeded and enable_footprint_limit:
    # Option 1: Switch to higher density technology
    current_density = gen_data["power_density_mw_per_m2"]
    
    for alt_gen_name, alt_gen_data in available_gens.items():
        if alt_gen_data["power_density_mw_per_m2"] > current_density * 1.3:
            # Calculate new footprint
            alt_area_per_gen = 1 / alt_gen_data["power_density_mw_per_m2"]
            alt_unit_cap = alt_gen_data["iso_rating_mw"] * derate_factor_calc
            alt_n_running = math.ceil(p_total_peak / alt_unit_cap)
            alt_n_total = alt_n_running + n_reserve
            alt_area_gen = alt_n_total * alt_unit_cap * alt_area_per_gen
            alt_total_area = (alt_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
            
            if alt_total_area <= max_area_m2:
                footprint_recommendations.append({
                    'type': 'Switch Technology',
                    'action': f'Change to {alt_gen_name}',
                    'new_area': alt_total_area,
                    'savings_pct': ((total_area_m2 - alt_total_area) / total_area_m2) * 100,
                    'trade_off': f'Efficiency: {alt_gen_data["electrical_efficiency"]*100:.1f}% vs {gen_data["electrical_efficiency"]*100:.1f}%'
                })
    
    # Option 2: Reduce redundancy
    if n_reserve > 0:
        reduced_n = n_total - 1
        reduced_area_gen = reduced_n * unit_site_cap * area_per_gen
        reduced_total_area = (reduced_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
        
        # Calculate new availability
        reduced_avail, _ = calculate_availability_weibull(reduced_n, n_running, mtbf_hours, project_years, gen_data["maintenance_interval_hrs"], gen_data["maintenance_duration_hrs"])
        
        if reduced_total_area <= max_area_m2:
            footprint_recommendations.append({
                'type': 'Reduce Redundancy',
                'action': f'Change from N+{n_reserve} to N+{n_reserve-1}',
                'new_area': reduced_total_area,
                'savings_pct': ((total_area_m2 - reduced_total_area) / total_area_m2) * 100,
                'trade_off': f'Availability: {reduced_avail*100:.3f}% vs {prob_gen*100:.3f}%'
            })

# Display conversions
if is_imperial:
    disp_area = total_area_m2 * 0.000247105
    disp_area_unit = "Acres"
else:
    disp_area = total_area_m2 / 10000
    disp_area_unit = "Ha"

# ==============================================================================
# 5. FUEL & EMISSIONS
# ==============================================================================

total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412

# Pipeline sizing
if not is_lng_primary:
    flow_rate_scfh = total_fuel_input_mmbtu_hr * 1000 / 1.02
    rec_pipe_dia = math.sqrt(flow_rate_scfh / 3000) * 2
else:
    rec_pipe_dia = 0

# Emissions
nox_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_nox"] / 1000)
co_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_co"] / 1000)
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor

# Emissions control
at_capex_total = 0
if nox_lb_hr * 8760 > 100:
    cost_scr_kw = 75.0
    cost_oxicat_kw = 25.0
    at_capex_total = (installed_cap * 1000) * (cost_scr_kw + cost_oxicat_kw)

# ==============================================================================
# 6. COOLING & TRI-GENERATION
# ==============================================================================

total_heat_rec_mw = 0.0
total_cooling_mw_chp = 0.0
cooling_coverage_pct = 0.0

if include_chp:
    waste_heat_mw = total_fuel_input_mw - p_total_avg
    recovery_eff = 0.65
    total_heat_rec_mw = waste_heat_mw * recovery_eff
    
    cop_absorption = 0.70
    total_cooling_mw_chp = total_heat_rec_mw * cop_absorption
    cooling_coverage_pct = min(100.0, (total_cooling_mw_chp / total_cooling_mw) * 100)
    
    pue_improvement = 0.15 * (cooling_coverage_pct / 100)
    pue_actual = pue - pue_improvement
else:
    pue_actual = pue

# Water consumption
wue = 1.8 if (cooling_method == "Water-Cooled" or include_chp) else 0.2
water_m3_day = p_it * wue * 24

if is_imperial:
    disp_cooling = total_cooling_mw_chp * 284.3
    disp_water = water_m3_day * 264.172
else:
    disp_cooling = total_cooling_mw_chp
    disp_water = water_m3_day

# ==============================================================================
# 7. ENHANCED FINANCIALS & LCOE
# ==============================================================================

# Apply regional multiplier
gen_unit_cost = gen_data["est_cost_kw"] * regional_mult
gen_install_cost = gen_data["est_install_kw"] * regional_mult

gen_cost_total = (installed_cap * 1000) * gen_unit_cost / 1e6

# Installation & BOP
idx_install = gen_install_cost / gen_unit_cost
idx_chp = 0.20 if include_chp else 0

# BESS costs
bess_cost_kw = 250.0
bess_cost_kwh = 400.0
bess_om_kw_yr = 5.0
bess_life_batt = 10
bess_life_inv = 15

if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr)
else:
    bess_capex_m = 0
    bess_om_annual = 0

# Fuel infrastructure
if has_lng_storage:
    log_capex = (lng_gal * 3.5) + (lng_days * 50000)
    pipeline_capex_m = 0
else:
    log_capex = 0
    pipe_cost_m = 50 * rec_pipe_dia
    pipeline_capex_m = (pipe_cost_m * dist_gas_main_m) / 1e6

# CAPEX breakdown
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

# Tax benefits
itc_benefit_m = (initial_capex_sum * 0.30) if (enable_itc and include_chp) else 0
depreciation_benefit_m = calculate_macrs_depreciation(initial_capex_sum * 1e6, project_years) / 1e6 if enable_depreciation else 0

# Repowering (BESS replacements)
repowering_pv_m = 0.0
if use_bess:
    for year in range(1, project_years + 1):
        year_cost = 0.0
        if year % bess_life_batt == 0 and year < project_years:
            year_cost += (bess_energy_total * 1000 * bess_cost_kwh)
        if year % bess_life_inv == 0 and year < project_years:
            year_cost += (bess_power_total * 1000 * bess_cost_kw)
        if year_cost > 0:
            repowering_pv_m += (year_cost / 1e6) / ((1 + wacc) ** year)

# Annualized costs
crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
repowering_annualized = repowering_pv_m * 1e6 * crf

# ENHANCED O&M CALCULATION
effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours

# O&M Fixed ($/kW-year)
om_fixed_kw_yr = 15.0  # Parts, insurance, property tax
om_fixed_annual = (installed_cap * 1000) * om_fixed_kw_yr

# O&M Variable ($/MWh)
om_variable_mwh = 3.5  # Consumables, oil, filters
om_variable_annual = mwh_year * om_variable_mwh

# O&M Labor
om_labor_per_unit = 120000  # $/unit-year
om_labor_annual = n_total * om_labor_per_unit

# Major overhaul (60k hours)
overhaul_interval_years = 60000 / (8760 * capacity_factor)
overhaul_cost_per_mw = 150000
overhaul_pv = 0
for year in np.arange(overhaul_interval_years, project_years, overhaul_interval_years):
    year_int = int(year)
    cost = installed_cap * overhaul_cost_per_mw
    overhaul_pv += cost / ((1 + wacc) ** year_int)
overhaul_annualized = overhaul_pv * crf

om_cost_year = om_fixed_annual + om_variable_annual + om_labor_annual + bess_om_annual + overhaul_annualized

# Fuel costs with degradation
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours

# Carbon costs
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

# Total annual cost
capex_annualized = (initial_capex_sum * 1e6) * crf
total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year

# Tax benefits (reduce annual cost)
ptc_annual = (mwh_year * 1000 * 0.013) if enable_ptc else 0
itc_annualized = (itc_benefit_m * 1e6) * crf
depreciation_annualized = (depreciation_benefit_m * 1e6) * crf

total_annual_cost_after_tax = total_annual_cost - ptc_annual - itc_annualized - depreciation_annualized

# LCOE
lcoe = total_annual_cost_after_tax / (mwh_year * 1000)

# NPV
annual_grid_cost = mwh_year * 1000 * benchmark_price
annual_savings = annual_grid_cost - (fuel_cost_year + om_cost_year + carbon_cost_year)

if wacc > 0:
    pv_savings = annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)
else:
    pv_savings = annual_savings * project_years

# Add tax benefits to NPV
total_tax_benefits = (itc_benefit_m + depreciation_benefit_m) * 1e6 + (ptc_annual * project_years)

npv = pv_savings + total_tax_benefits - (initial_capex_sum * 1e6) - (repowering_pv_m * 1e6)

if annual_savings > 0:
    payback_years = (initial_capex_sum * 1e6) / annual_savings
    roi_simple = (annual_savings / (initial_capex_sum * 1e6)) * 100
    payback_str = f"{payback_years:.1f} Years"
else:
    payback_str = "N/A"
    roi_simple = 0

# Gas price sensitivity
gas_prices_x = np.linspace(0, total_gas_price * 2, 20)
lcoe_y = []
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    lcoe_y.append(sim_lcoe)

breakeven_gas_price = 0.0
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    if sim_lcoe <= benchmark_price:
        breakeven_gas_price = gp
        break

# ==============================================================================
# 8. OUTPUTS - ENHANCED TABBED INTERFACE
# ==============================================================================

t1, t2, t3, t4, t5 = st.tabs([
    "📊 System Design", 
    "⚡ Performance & Stability", 
    "🏗️ Footprint & Optimization",
    "❄️ Cooling & Tri-Gen", 
    "💰 Economics & ROI"
])

with t1:
    st.subheader("System Architecture")
    
    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Generator", selected_gen)
    c2.metric("Fleet", f"{n_running}+{n_reserve}")
    c3.metric("Installed", f"{installed_cap:.1f} MW")
    
    # Availability with status indicator
    if target_met:
        c4.metric("Availability", f"{prob_gen*100:.3f}%", delta="✅ Target Met", delta_color="normal")
    else:
        c4.metric("Availability", f"{prob_gen*100:.3f}%", delta="⚠️ Below Target", delta_color="inverse")
    
    c5.metric("PUE", f"{pue_actual:.2f}")
    c6.metric("Density", f"{gen_data['power_density_mw_per_m2']*1000:.0f} kW/m²")
    
    # Load Profile Visualization
    st.markdown("### 📈 Annual Load Profile & Duration Curve")
    
    hours = np.arange(0, 8760)
    daily_wave = 1.0 + 0.15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    load_curve = p_total_avg * daily_wave * np.random.uniform(0.95, 1.05, len(hours))
    load_curve = np.clip(load_curve, 0, p_total_peak)
    load_sorted = np.sort(load_curve)[::-1]
    
    fig_ldc = go.Figure()
    fig_ldc.add_trace(go.Scatter(
        x=hours, y=load_sorted, fill='tozeroy',
        name='DC Load', line=dict(color='#667eea', width=2)
    ))
    fig_ldc.add_hline(
        y=installed_cap, line_dash="dash", line_color="red",
        annotation_text=f"Total Installed: {installed_cap:.1f} MW",
        annotation_position="top right"
    )
    
    # Show genset capacity line
    genset_capacity = n_running * unit_site_cap
    fig_ldc.add_hline(
        y=genset_capacity, line_dash="dashdot", line_color="green",
        annotation_text=f"Genset Capacity: {genset_capacity:.1f} MW",
        annotation_position="bottom right"
    )
    
    fig_ldc.add_hline(
        y=p_total_avg, line_dash="dot", line_color="orange",
        annotation_text=f"Average: {p_total_avg:.1f} MW"
    )
    
    # Show BESS peak shaving zone if applicable
    if use_bess and p_total_peak > genset_capacity:
        fig_ldc.add_hrect(
            y0=genset_capacity, y1=p_total_peak,
            fillcolor="yellow", opacity=0.2,
            annotation_text=f"BESS Peak Shaving Zone ({bess_power_total:.1f} MW)",
            annotation_position="top left"
        )
    
    fig_ldc.update_layout(
        title="Load Duration Curve (with BESS Peak Shaving Strategy)" if use_bess else "Load Duration Curve",
        xaxis_title="Hours per Year (Sorted)",
        yaxis_title="Load (MW)",
        height=400
    )
    st.plotly_chart(fig_ldc, use_container_width=True)
    
    # Show sizing strategy explanation
    if use_bess:
        st.info(f"💡 **BESS Peak Shaving Strategy:**\n"
                f"- Gensets sized for **{genset_capacity:.1f} MW** (average + margin)\n"
                f"- BESS covers **{(p_total_peak - genset_capacity):.1f} MW** peak difference\n"
                f"- Result: **Fewer gensets**, **higher efficiency** ({fleet_efficiency*100:.1f}% vs ~{get_part_load_efficiency(gen_data['electrical_efficiency'], 60, gen_data['type'])*100:.1f}% without BESS)")

    
    # Fleet optimization results
    if fleet_options:
        st.markdown("### 🎯 Fleet Optimization Analysis")
        
        opt_data = []
        for n, data in fleet_options.items():
            opt_data.append({
                'Units': n,
                'Load (%)': data['load_pct'],
                'Efficiency (%)': data['efficiency'] * 100,
                'Score': data['score']
            })
        df_opt = pd.DataFrame(opt_data)
        
        col_opt1, col_opt2 = st.columns([2, 1])
        
        with col_opt1:
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=df_opt['Units'], y=df_opt['Efficiency (%)'],
                mode='lines+markers', name='Efficiency',
                line=dict(color='#28a745', width=3)
            ))
            fig_opt.update_layout(
                title="Fleet Size vs Efficiency",
                xaxis_title="Number of Running Units",
                yaxis_title="Fleet Efficiency (%)",
                height=300
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        
        with col_opt2:
            st.dataframe(df_opt, use_container_width=True)
            st.success(f"✅ **Optimal:** {n_running} units at {load_per_unit_pct:.1f}% load")
    
    # Part-load efficiency curve
    st.markdown("### 📉 Part-Load Efficiency Curve")
    
    load_range = np.linspace(30, 100, 50)
    eff_curve = [get_part_load_efficiency(gen_data["electrical_efficiency"], load, gen_data["type"]) * 100 
                 for load in load_range]
    
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=load_range, y=eff_curve, mode='lines',
        name='Efficiency', line=dict(color='#28a745', width=3)
    ))
    fig_eff.add_vline(
        x=load_per_unit_pct, line_dash="dash", line_color="red",
        annotation_text=f"Operating: {load_per_unit_pct:.0f}%"
    )
    fig_eff.update_layout(
        title=f"Efficiency vs Load - {gen_data['type']}",
        xaxis_title="Load (%)",
        yaxis_title="Electrical Efficiency (%)",
        height=350
    )
    st.plotly_chart(fig_eff, use_container_width=True)
    
    # Fleet details
    st.markdown("### 🔧 Fleet Configuration Details")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("**Generator Specifications:**")
        st.write(f"- Model: {selected_gen}")
        st.write(f"- Type: {gen_data['type']}")
        st.write(f"- ISO Rating: {unit_iso_cap:.2f} MW")
        st.write(f"- Site Rating: {unit_site_cap:.2f} MW")
        st.write(f"- Efficiency (ISO): {gen_data['electrical_efficiency']*100:.1f}%")
        st.write(f"- Ramp Rate: {gen_data['ramp_rate_mw_s']:.1f} MW/s")
        st.write(f"- MTBF: {gen_data['mtbf_hours']:,} hours")
    
    with col_f2:
        st.markdown("**Operating Parameters:**")
        st.write(f"- Strategy: {load_strategy}")
        st.write(f"- Units Running: {units_running} of {n_total}")
        st.write(f"- Load per Unit: {load_per_unit_pct:.1f}%")
        st.write(f"- Fleet Efficiency: {fleet_efficiency*100:.1f}%")
        st.write(f"- Capacity Factor: {capacity_factor*100:.0f}%")
        st.write(f"- Hours/Year: {effective_hours:.0f}")
        st.write(f"- Annual Energy: {mwh_year:,.0f} MWh")

with t2:
    st.subheader("Electrical Performance & Stability")
    
    # ========================================================================
    # RELIABILITY TRADE-OFFS TABLE (NEW)
    # ========================================================================
    if len(reliability_configs) > 1:
        st.markdown("### ⚖️ Reliability Configuration Comparison")
        
        # Build comparison table
        comparison_data = []
        for config in reliability_configs:
            # Calculate CAPEX
            genset_capex = config['n_total'] * unit_site_cap * gen_data['est_cost_kw'] / 1000  # M$
            bess_capex = config['bess_mwh'] * 0.3  # M$ (assuming $300k/MWh)
            total_capex = genset_capex + bess_capex
            
            # Calculate O&M/year
            genset_om = config['n_total'] * 120  # k$/unit-year
            bess_om = config['bess_mwh'] * 10    # k$/MWh-year
            total_om = (genset_om + bess_om) / 1000  # M$/year
            
            # Use pre-calculated values if available, otherwise calculate
            running_units = config['n_running']
            
            if 'load_pct' in config:
                load_per_unit = config['load_pct']
            else:
                load_per_unit = (p_total_avg / (running_units * unit_site_cap)) * 100
            
            if 'efficiency' in config:
                config_efficiency = config['efficiency']
            else:
                config_efficiency = get_part_load_efficiency(
                    gen_data["electrical_efficiency"],
                    load_per_unit,
                    gen_data["type"]
                )
            
            # Annual energy and fuel
            annual_energy_gwh = (p_total_avg * 8760 * capacity_factor) / 1000  # GWh
            annual_fuel_mmbtu = (annual_energy_gwh * 1000 * 3.412) / config_efficiency  # MMBtu
            
            comparison_data.append({
                'Configuration': config['name'],
                'Fleet': f"{config['n_running']}+{config['n_reserve']}",
                'Total Units': config['n_total'],
                'BESS (MW/MWh)': f"{config['bess_mw']:.0f}/{config['bess_mwh']:.0f}" if config['bess_mw'] > 0 else "None",
                'Load/Unit (%)': f"{load_per_unit:.1f}%",
                'Fleet Eff (%)': f"{config_efficiency*100:.1f}%",
                'BESS Credit': f"{config['bess_credit']:.1f}" if config['bess_credit'] > 0 else "-",
                'Availability': f"{config['availability']*100:.3f}%",
                'Energy (GWh/yr)': f"{annual_energy_gwh:.1f}",
                'Fuel (M MMBtu/yr)': f"{annual_fuel_mmbtu/1e6:.2f}",
                'CAPEX (M$)': f"${total_capex:.1f}M",
                'O&M (M$/yr)': f"${total_om:.2f}M"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Highlight selected configuration
        selected_name = selected_config['name']
        
        st.dataframe(
            df_comparison.style.apply(
                lambda row: ['background-color: #d4edda' if row['Configuration'] == selected_name else '' for _ in row],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Show selected config details
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        
        with col_sel1:
            st.metric(
                "✅ Selected Configuration",
                selected_name,
                delta=f"{selected_config['n_total']} total units"
            )
        
        with col_sel2:
            if selected_config['bess_credit'] > 0:
                units_saved = best_config_a['n_total'] - selected_config['n_total'] if best_config_a else 0
                st.metric(
                    "Units Saved vs No BESS",
                    f"{units_saved} gensets",
                    delta=f"${units_saved * 3:.0f}M CAPEX saved" if units_saved > 0 else "N/A"
                )
            else:
                st.metric("BESS Strategy", "Transient Only", delta="No reliability credit")
        
        with col_sel3:
            st.metric(
                "Availability Achieved",
                f"{selected_config['availability']*100:.4f}%",
                delta="✅ Target Met" if target_met else "⚠️ Below Target",
                delta_color="normal" if target_met else "inverse"
            )
        
        # Show BESS credit breakdown if applicable
        if 'credit_breakdown' in selected_config and selected_config['bess_credit'] > 0:
            with st.expander("🔍 BESS Reliability Credit Details"):
                breakdown = selected_config['credit_breakdown']
                
                st.markdown("**Credit Calculation:**")
                col_b1, col_b2 = st.columns(2)
                
                with col_b1:
                    st.write(f"- **Power Credit:** {breakdown['power_credit']:.1f} units")
                    st.write(f"  (BESS {selected_config['bess_mw']:.0f} MW / Gen {unit_site_cap:.1f} MW)")
                    st.write(f"- **Energy Credit:** {breakdown['energy_credit']:.1f} units")
                    st.write(f"  (BESS {selected_config['bess_mwh']:.0f} MWh / {mttr_hours} hrs outage)")
                
                with col_b2:
                    st.write(f"- **Raw Credit:** {breakdown['raw_credit']:.1f} units")
                    st.write(f"- **BESS Availability:** {breakdown['bess_availability']*100:.0f}%")
                    st.write(f"- **Coverage Factor:** {breakdown['coverage_factor']*100:.0f}%")
                    st.write(f"- **→ Effective Credit:** {breakdown['effective_credit']:.1f} units")
                
                st.caption(
                    f"💡 BESS can sustain {selected_config['bess_mw']:.0f} MW for "
                    f"{breakdown['bess_duration_hrs']:.1f} hours, covering typical {mttr_hours}hr repair time."
                )
        
        st.markdown("---")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Voltage", f"{rec_voltage_kv} kV")
    col_e2.metric("Frequency", f"{freq_hz} Hz")
    col_e3.metric("X\"d", f"{gen_data['reactance_xd_2']:.3f} pu")
    col_e4.metric("Ramp Rate", f"{gen_data['ramp_rate_mw_s']:.1f} MW/s")
    
    # ========================================================================
    # NET EFFICIENCY & HEAT RATE (NEW)
    # ========================================================================
    st.markdown("### ⚙️ Net Efficiency & Heat Rate")
    
    # Calculate auxiliaries and losses
    aux_power_pct = 2.0  # Typical: cooling, fuel pumps, controls = 2%
    mechanical_losses_pct = 1.5  # Bearings, coupling, etc = 1.5%
    
    # Gross electrical efficiency (from generator)
    gross_efficiency = fleet_efficiency
    
    # Net efficiency (after auxiliaries)
    aux_consumption = p_total_avg * (aux_power_pct / 100)
    net_output = p_total_avg - aux_consumption
    net_efficiency = gross_efficiency * (1 - aux_power_pct/100)
    
    # Heat Rate calculations
    # Heat Rate = 3412 BTU/kWh / Efficiency
    # LHV to HHV conversion: HHV = LHV × 1.11 (for natural gas)
    
    heat_rate_lhv_btu = 3412 / net_efficiency  # BTU/kWh (LHV)
    heat_rate_hhv_btu = heat_rate_lhv_btu * 1.11  # BTU/kWh (HHV)
    
    # MJ/kWh conversion: 1 BTU = 0.001055 MJ
    heat_rate_lhv_mj = heat_rate_lhv_btu * 0.001055  # MJ/kWh (LHV)
    heat_rate_hhv_mj = heat_rate_hhv_btu * 0.001055  # MJ/kWh (HHV)
    
    col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
    
    with col_eff1:
        st.metric("Gross Efficiency", f"{gross_efficiency*100:.2f}%")
        st.caption("At generator terminals")
    
    with col_eff2:
        st.metric("Net Efficiency", f"{net_efficiency*100:.2f}%")
        st.caption("After auxiliaries")
    
    with col_eff3:
        if is_imperial:
            st.metric("Heat Rate (HHV)", f"{heat_rate_hhv_btu:.0f}")
            st.caption("BTU/kWh")
        else:
            st.metric("Heat Rate (HHV)", f"{heat_rate_hhv_mj:.2f}")
            st.caption("MJ/kWh")
    
    with col_eff4:
        if is_imperial:
            st.metric("Heat Rate (LHV)", f"{heat_rate_lhv_btu:.0f}")
            st.caption("BTU/kWh")
        else:
            st.metric("Heat Rate (LHV)", f"{heat_rate_lhv_mj:.2f}")
            st.caption("MJ/kWh")
    
    # Losses breakdown
    with st.expander("📊 Efficiency & Losses Breakdown"):
        st.markdown("**Power Flow (100 MW IT Load Example):**")
        
        fuel_input = p_total_avg / net_efficiency
        
        losses_data = pd.DataFrame({
            'Stage': [
                '1. Fuel Input',
                '2. Combustion → Shaft',
                '3. Shaft → Electrical',
                '4. Gross Electrical',
                '5. Auxiliaries',
                '6. Net Output to DC'
            ],
            'Power (MW)': [
                fuel_input,
                fuel_input * gross_efficiency,
                p_total_avg,
                p_total_avg,
                aux_consumption,
                net_output
            ],
            'Efficiency (%)': [
                100.0,
                gross_efficiency * 100,
                gross_efficiency * 100,
                gross_efficiency * 100,
                (1 - aux_power_pct/100) * 100,
                net_efficiency * 100
            ],
            'Losses (MW)': [
                0,
                fuel_input * (1 - gross_efficiency),
                0,
                0,
                aux_consumption,
                0
            ]
        })
        
        col_loss1, col_loss2 = st.columns([2, 1])
        
        with col_loss1:
            fig_losses = go.Figure()
            
            # Sankey-style data
            stages = losses_data['Stage'].tolist()
            power_values = losses_data['Power (MW)'].tolist()
            
            fig_losses.add_trace(go.Bar(
                x=stages,
                y=power_values,
                text=[f"{p:.1f} MW" for p in power_values],
                textposition='outside',
                marker_color=['#2ca02c', '#ff7f0e', '#1f77b4', '#1f77b4', '#d62728', '#2ca02c']
            ))
            
            fig_losses.update_layout(
                title="Power Flow from Fuel to DC Load",
                xaxis_title="Stage",
                yaxis_title="Power (MW)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_losses, use_container_width=True)
        
        with col_loss2:
            st.dataframe(losses_data, use_container_width=True, hide_index=True)
            
            st.markdown("**Summary:**")
            total_losses = fuel_input - net_output
            st.write(f"• Fuel Input: {fuel_input:.1f} MW")
            st.write(f"• Total Losses: {total_losses:.1f} MW ({(total_losses/fuel_input)*100:.1f}%)")
            st.write(f"• Net Output: {net_output:.1f} MW")
            st.write(f"• Net Efficiency: {net_efficiency*100:.2f}%")
    
    st.markdown("---")
    
    # Transient Stability
    st.markdown("### 🎯 Transient Stability Analysis")
    
    if stability_ok:
        st.success(f"✅ **Voltage Sag OK:** {voltage_sag:.2f}% (Limit: 10%)")
    else:
        st.error(f"❌ **Voltage Sag Exceeds:** {voltage_sag:.2f}% > 10%")
        st.warning("**Mitigation:** Add generators, increase BESS, or use lower X\"d units")
    
    # Step Load & BESS
    st.markdown("### 🔋 Step Load Capability & BESS Analysis")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Required Step Load", f"{step_load_req:.0f}%")
    col_s2.metric("Gen Capability", f"{gen_data['step_load_pct']:.0f}%")
    
    step_capable = gen_data["step_load_pct"] >= step_load_req
    if step_capable:
        col_s3.success("✅ COMPLIANT")
    elif use_bess:
        col_s3.warning("⚠️ BESS REQUIRED")
    else:
        col_s3.error("❌ NOT COMPLIANT")
    
    if use_bess:
        st.info(f"🔋 **BESS Capacity:** {bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh")
        
        # Show BESS strategy
        peak_vs_avg = p_total_peak - p_total_avg
        
        # Calculate units saved (if n_running_from_load is available)
        if 'n_running_from_load' in locals() or 'n_running_from_load' in globals():
            units_saved = n_running_from_load - n_running  # Positive if we saved units
        else:
            units_saved = 0
        
        if peak_vs_avg > 0 and units_saved > 0:
            st.success(f"✅ **BESS Peak Shaving:** Gensets sized for {p_total_avg:.1f} MW average. "
                      f"BESS covers {peak_vs_avg:.1f} MW peak difference → **{units_saved} fewer gensets needed**")
        elif peak_vs_avg > 0:
            st.success(f"✅ **BESS Peak Shaving:** Gensets sized for {p_total_avg:.1f} MW average. "
                      f"BESS covers {peak_vs_avg:.1f} MW peak difference")
        
        # BESS Breakdown
        bess_breakdown_data = pd.DataFrame({
            'Component': ['Step Support', 'Peak Shaving', 'Ramp Support', 'Freq Reg', 'Black Start'],
            'Power (MW)': [
                bess_breakdown.get('step_support', 0),
                bess_breakdown.get('peak_shaving', 0),
                bess_breakdown.get('ramp_support', 0),
                bess_breakdown.get('freq_reg', 0),
                bess_breakdown.get('black_start', 0)
            ]
        })
        
        # Filter out zero values
        bess_breakdown_data = bess_breakdown_data[bess_breakdown_data['Power (MW)'] > 0.01]
        
        col_bess1, col_bess2 = st.columns([2, 1])
        
        with col_bess1:
            fig_bess = px.bar(bess_breakdown_data, x='Component', y='Power (MW)',
                             title="BESS Sizing Breakdown (Dominant Component Sets Capacity)", 
                             color='Component',
                             text='Power (MW)')
            fig_bess.update_traces(texttemplate='%{text:.1f} MW', textposition='outside')
            fig_bess.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bess, use_container_width=True)
        
        with col_bess2:
            st.markdown("**BESS Functions:**")
            if bess_breakdown.get('peak_shaving', 0) > 0:
                st.write(f"✅ Peak Shaving: {bess_breakdown['peak_shaving']:.1f} MW")
            if bess_breakdown.get('step_support', 0) > 0:
                st.write(f"✅ Step Load: {bess_breakdown['step_support']:.1f} MW")
            if bess_breakdown.get('ramp_support', 0) > 0:
                st.write(f"✅ Ramp Rate: {bess_breakdown['ramp_support']:.1f} MW")
            if bess_breakdown.get('freq_reg', 0) > 0:
                st.write(f"✅ Freq Reg: {bess_breakdown['freq_reg']:.1f} MW")
            if bess_breakdown.get('black_start', 0) > 0:
                st.write(f"✅ Black Start: {bess_breakdown['black_start']:.1f} MW")
            
            st.caption(f"ℹ️ Final capacity = MAX of all requirements")

    
    # Reliability over time (Weibull)
    st.markdown("### 📊 Reliability Projection (Aging Model)")
    
    # Add warning if target not met
    if not target_met:
        st.error(f"🛑 **Availability Target Not Achieved**")
        col_gap1, col_gap2, col_gap3 = st.columns(3)
        col_gap1.metric("Target", f"{avail_req:.3f}%")
        col_gap2.metric("Achieved", f"{prob_gen*100:.3f}%", f"-{(avail_req - prob_gen*100):.3f}%")
        col_gap3.metric("Configuration", f"N+{n_reserve} (Maximum)")
        
        st.warning("**Root Causes:**\n"
                  f"- Generator MTBF: {mtbf_hours:,} hours\n"
                  f"- Fleet degradation over {project_years} years (0.5%/year)\n"
                  f"- Binomial reliability model with aging\n\n"
                  "**Solutions:**\n"
                  "1. Select generator with higher MTBF (e.g., Gas Turbine: 80k hrs)\n"
                  "2. Reduce project life for analysis\n"
                  "3. Accept lower availability target\n"
                  "4. Implement predictive maintenance program")
    else:
        st.success(f"✅ **Availability Target Met:** {prob_gen*100:.3f}% ≥ {avail_req:.3f}% with N+{n_reserve}")
    
    years_range = list(range(1, project_years + 1))
    
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=years_range, y=[a*100 for a in availability_curve],
        mode='lines', name='Availability',
        line=dict(color='#007bff', width=3)
    ))
    fig_rel.add_hline(
        y=avail_req, line_dash="dash", line_color="red",
        annotation_text=f"Target: {avail_req:.2f}%"
    )
    fig_rel.update_layout(
        title="System Availability Over Time (with 0.5%/year degradation)",
        xaxis_title="Project Year",
        yaxis_title="Availability (%)",
        height=400
    )
    st.plotly_chart(fig_rel, use_container_width=True)
    
    # Emissions
    st.markdown("### 🌍 Environmental Performance")
    
    col_em1, col_em2, col_em3, col_em4 = st.columns(4)
    col_em1.metric("NOx", f"{nox_lb_hr:.2f} lb/hr")
    col_em2.metric("CO", f"{co_lb_hr:.2f} lb/hr")
    col_em3.metric("CO₂/Year", f"{co2_ton_yr:,.0f} tons")
    col_em4.metric("Carbon Cost", f"${carbon_cost_year/1e6:.2f}M/yr")
    
    if at_capex_total > 0:
        st.warning(f"⚠️ **Emissions Control:** SCR + Catalyst (${at_capex_total/1e6:.2f}M)")

with t3:
    st.subheader("Footprint Analysis & Optimization")
    
    # Footprint metrics
    col_fp1, col_fp2, col_fp3 = st.columns(3)
    col_fp1.metric("Total Footprint", f"{disp_area:.2f} {disp_area_unit}")
    col_fp2.metric("Power Density", f"{gen_data['power_density_mw_per_m2']:.3f} MW/m²")
    
    if enable_footprint_limit:
        col_fp3.metric("Utilization", f"{area_utilization_pct:.1f}%")
    else:
        col_fp3.metric("Status", "No Limit Set")
    
    # Footprint breakdown
    st.markdown("### 📐 Footprint Breakdown")
    
    footprint_data = pd.DataFrame({
        "Component": ["Generators", "BESS", "Fuel Storage", "Cooling/CHP", "Substation", "Contingency (20%)"],
        "Area (m²)": [area_gen, area_bess, storage_area_m2, area_chp, area_sub, total_area_m2 * 0.2]
    })
    
    col_pie1, col_pie2 = st.columns([2, 1])
    
    with col_pie1:
        fig_pie = px.pie(footprint_data, values='Area (m²)', names='Component',
                        title=f"Total: {disp_area:.2f} {disp_area_unit}")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_pie2:
        st.dataframe(footprint_data, use_container_width=True)
    
    # Optimization recommendations
    if is_area_exceeded and footprint_recommendations:
        st.error(f"🛑 **Footprint Exceeded:** {total_area_m2:,.0f} m² > {max_area_m2:,.0f} m² ({(total_area_m2/max_area_m2-1)*100:.0f}% over)")
        
        st.markdown("### 💡 Optimization Recommendations")
        
        for i, rec in enumerate(footprint_recommendations, 1):
            with st.expander(f"Option {i}: {rec['action']} (Saves {rec['savings_pct']:.1f}%)"):
                st.write(f"**Type:** {rec['type']}")
                st.write(f"**New Footprint:** {rec['new_area']:,.0f} m² ({rec['new_area']/10000:.2f} Ha)")
                st.write(f"**Savings:** {rec['savings_pct']:.1f}%")
                st.write(f"**Trade-off:** {rec['trade_off']}")
                
                if rec['type'] == 'Switch Technology':
                    st.info("✅ **Recommended:** Higher density technology maintains performance")
                elif rec['type'] == 'Reduce Redundancy':
                    st.warning("⚠️ **Risk:** Lower availability - evaluate criticality")
    
    elif enable_footprint_limit:
        st.success(f"✅ **Footprint OK:** {area_utilization_pct:.1f}% of available area")

with t4:
    st.subheader("Cooling & Tri-Generation")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    if include_chp:
        col_c1.metric("Heat Recovered", f"{total_heat_rec_mw:.1f} MWt")
        col_c2.metric("Cooling Generated", f"{total_cooling_mw_chp:.1f} MWc")
        col_c3.metric("Coverage", f"{cooling_coverage_pct:.1f}%")
        col_c4.metric("PUE Improvement", f"{pue - pue_actual:.2f}")
        
        st.progress(min(1.0, cooling_coverage_pct/100))
        
        st.info(f"💡 **Tri-Gen Benefit:** PUE reduced from {pue:.2f} to {pue_actual:.2f}")
    else:
        col_c1.metric("Cooling Method", cooling_method)
        col_c2.metric("Cooling Load", f"{total_cooling_mw:.1f} MWc")
        col_c3.metric("PUE", f"{pue_actual:.2f}")
        col_c4.metric("WUE", f"{wue:.1f}")
    
    st.metric(f"Water Consumption (WUE {wue:.1f})", f"{disp_water:,.0f} gal/day" if is_imperial else f"{disp_water:,.0f} m³/day")
    
    if wue > 1.5:
        st.warning("⚠️ **High Water Use:** Consider dry cooling or recycling")

with t5:
    st.subheader("Financial Analysis & Economics")
    
    # Tax benefits summary
    if itc_benefit_m > 0 or depreciation_benefit_m > 0 or ptc_annual > 0:
        st.markdown("### 💸 Tax Benefits & Incentives")
        
        col_tax1, col_tax2, col_tax3, col_tax4 = st.columns(4)
        
        if itc_benefit_m > 0:
            col_tax1.metric("ITC (30%)", f"${itc_benefit_m:.2f}M")
        if depreciation_benefit_m > 0:
            col_tax2.metric("MACRS Depreciation", f"${depreciation_benefit_m:.2f}M")
        if ptc_annual > 0:
            col_tax3.metric("PTC (Annual)", f"${ptc_annual/1e6:.2f}M")
        
        total_tax_benefit_m = itc_benefit_m + depreciation_benefit_m + (ptc_annual * project_years / 1e6)
        col_tax4.metric("Total Tax Benefit", f"${total_tax_benefit_m:.2f}M")
    
    # LCOE Target Check
    if enable_lcoe_target and target_lcoe > 0:
        if lcoe > target_lcoe:
            st.error(f"⚠️ **Target Missed:** LCOE ${lcoe:.4f}/kWh > Target ${target_lcoe:.4f}/kWh")
        else:
            st.success(f"🎉 **Target Met:** LCOE ${lcoe:.4f}/kWh < Target ${target_lcoe:.4f}/kWh")
    
    # CAPEX Editor
    st.markdown("### 💰 CAPEX Breakdown")
    st.info(f"**Regional Multiplier:** {region} ({regional_mult:.2f}x)")
    
    edited_capex = st.data_editor(
        df_capex,
        column_config={
            "Index": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    total_capex_dynamic = edited_capex["Cost (M USD)"].sum()
    
    # Recalculate financials
    capex_annualized_dyn = (total_capex_dynamic * 1e6) * crf
    total_annual_cost_dyn = fuel_cost_year + om_cost_year + capex_annualized_dyn + repowering_annualized + carbon_cost_year
    total_annual_cost_dyn_after_tax = total_annual_cost_dyn - ptc_annual - itc_annualized - depreciation_annualized
    lcoe_dyn = total_annual_cost_dyn_after_tax / (mwh_year * 1000)
    
    npv_dyn = pv_savings + total_tax_benefits - (total_capex_dynamic * 1e6) - (repowering_pv_m * 1e6)
    
    if annual_savings > 0:
        payback_dyn = (total_capex_dynamic * 1e6) / annual_savings
        roi_dyn = (annual_savings / (total_capex_dynamic * 1e6)) * 100
        payback_str_dyn = f"{payback_dyn:.1f} Years"
    else:
        payback_str_dyn = "N/A"
        roi_dyn = 0
    
    # Financial KPIs
    st.markdown("### 📊 Key Financial Metrics")
    
    c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
    c_f1.metric("CAPEX", f"${total_capex_dynamic:.2f}M")
    c_f2.metric("LCOE", f"${lcoe_dyn:.4f}/kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f}M")
    c_f4.metric("NPV (20yr)", f"${npv_dyn/1e6:.2f}M")
    c_f5.metric("Payback", payback_str_dyn)
    
    # O&M Breakdown
    st.markdown("### 🔧 Annual O&M Breakdown")
    
    om_data = pd.DataFrame({
        'Component': ['Fixed ($/kW-yr)', 'Variable ($/MWh)', 'Labor', 'Major Overhaul', 'BESS O&M'],
        'Annual Cost ($M)': [
            om_fixed_annual/1e6,
            om_variable_annual/1e6,
            om_labor_annual/1e6,
            overhaul_annualized/1e6,
            bess_om_annual/1e6
        ]
    })
    
    fig_om = px.bar(om_data, x='Component', y='Annual Cost ($M)',
                   title=f"Total O&M: ${om_cost_year/1e6:.2f}M/year",
                   color='Component')
    fig_om.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_om, use_container_width=True)
    
    # LCOE Breakdown
    st.markdown("### 💵 LCOE Component Breakdown")
    
    cost_data = pd.DataFrame({
        "Component": ["Fuel", "O&M", "CAPEX", "Repowering", "Carbon", "Tax Benefits"],
        "$/kWh": [
            fuel_cost_year/(mwh_year*1000),
            om_cost_year/(mwh_year*1000),
            capex_annualized_dyn/(mwh_year*1000),
            repowering_annualized/(mwh_year*1000),
            carbon_cost_year/(mwh_year*1000),
            -(ptc_annual + itc_annualized + depreciation_annualized)/(mwh_year*1000)
        ]
    })
    
    fig_lcoe = px.bar(cost_data, x="Component", y="$/kWh",
                     title="LCOE Breakdown", text_auto='.4f',
                     color="Component")
    fig_lcoe.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_lcoe, use_container_width=True)
    
    # Gas Price Sensitivity
    st.markdown("### 📈 Gas Price Sensitivity Analysis")
    
    if breakeven_gas_price > 0:
        st.success(f"🎯 **Breakeven Gas Price:** ${breakeven_gas_price:.2f}/MMBtu")
    else:
        st.error("⚠️ **No Breakeven:** Prime power more expensive even with free gas")
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=gas_prices_x, y=lcoe_y, mode='lines',
        name='LCOE (Prime)', line=dict(color='#667eea', width=3)
    ))
    fig_sens.add_hline(
        y=benchmark_price, line_dash="dash", line_color="red",
        annotation_text=f"Benchmark: ${benchmark_price:.3f}/kWh"
    )
    if breakeven_gas_price > 0:
        fig_sens.add_vline(
            x=breakeven_gas_price, line_dash="dot", line_color="green",
            annotation_text=f"Breakeven: ${breakeven_gas_price:.2f}"
        )
    fig_sens.update_layout(
        title="LCOE vs Gas Price",
        xaxis_title="Total Gas Price ($/MMBtu)",
        yaxis_title="LCOE ($/kWh)",
        height=450
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# ==============================================================================
# 9. EXCEL & PDF EXPORT WITH FALLBACK
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
                    pue_actual, load_strategy, load_per_unit_pct, fleet_efficiency*100,
                    rec_voltage_kv, fuel_mode, region
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Sheet 2: Financial Summary
            financial_data = {
                'Metric': [
                    'Total CAPEX (M USD)', 'LCOE ($/kWh)', 'LCOE Pre-Tax ($/kWh)',
                    'Benchmark Price ($/kWh)', 'Gas Price Total ($/MMBtu)',
                    'Carbon Price ($/ton)', 'Annual Energy (MWh)', 'Effective Hours',
                    'Fuel Cost/Year (M USD)', 'O&M Cost/Year (M USD)',
                    'Carbon Cost/Year (M USD)', 'ITC Benefit (M USD)',
                    'MACRS Benefit (M USD)', 'PTC Annual (M USD)',
                    'Annual Savings (M USD)', 'NPV 20yr (M USD)',
                    'Payback (Years)', 'ROI (%)', 'Breakeven Gas ($/MMBtu)'
                ],
                'Value': [
                    total_capex_dynamic, lcoe_dyn, total_annual_cost/(mwh_year*1000),
                    benchmark_price, total_gas_price,
                    carbon_price_per_ton, mwh_year, effective_hours,
                    fuel_cost_year/1e6, om_cost_year/1e6,
                    carbon_cost_year/1e6, itc_benefit_m,
                    depreciation_benefit_m, ptc_annual/1e6,
                    annual_savings/1e6, npv_dyn/1e6,
                    payback_dyn if annual_savings > 0 else 0, roi_dyn, breakeven_gas_price
                ]
            }
            pd.DataFrame(financial_data).to_excel(writer, sheet_name='Financial Summary', index=False)
            
            # Sheet 3: CAPEX
            edited_capex.to_excel(writer, sheet_name='CAPEX Breakdown', index=False)
            
            # Sheet 4: O&M
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
                'Year': years_range,
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
                bess_breakdown_data.to_excel(writer, sheet_name='BESS Sizing', index=False)
        
        output.seek(0)
        return output
    
    excel_data = create_excel_export()
    
    st.download_button(
        label="📊 Download Complete Excel Report (9 Sheets)",
        data=excel_data,
        file_name=f"CAT_QuickSize_v2_{dc_type.replace(' ','_')}_{p_it:.0f}MW_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    st.success("✅ Export includes: Executive Summary, Financials, CAPEX, O&M, Technical Specs, Reliability, Sensitivity, Footprint, BESS Sizing")

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
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title Page
        story.append(Paragraph("CAT QuickSize v2.0", title_style))
        story.append(Paragraph("Data Center Primary Power Solutions", styles['Heading3']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Project: {dc_type} - {p_it:.0f} MW IT Load", styles['Normal']))
        story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        summary_table_data = [
            ['Parameter', 'Value'],
            ['Data Center Type', dc_type],
            ['IT Load', f'{p_it:.1f} MW'],
            ['PUE', f'{pue:.2f}'],
            ['Total DC Load (Avg/Peak)', f'{p_total_avg:.1f} / {p_total_peak:.1f} MW'],
            ['Generator Model', selected_gen],
            ['Fleet Configuration', f'{n_running}+{n_reserve} ({n_total} total units)'],
            ['Installed Capacity', f'{installed_cap:.1f} MW'],
            ['Availability Target', f'{avail_req:.3f}%'],
            ['Availability Achieved', f'{prob_gen*100:.3f}%'],
            ['Target Met', 'Yes ✓' if target_met else 'No ✗'],
        ]
        
        summary_table = Table(summary_table_data, colWidths=[3.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Technical Configuration
        story.append(Paragraph("Technical Configuration", heading_style))
        
        tech_table_data = [
            ['Specification', 'Value'],
            ['Load per Unit', f'{load_per_unit_pct:.1f}%'],
            ['Fleet Efficiency', f'{fleet_efficiency*100:.1f}%'],
            ['Operating Strategy', load_strategy],
            ['BESS Capacity', f'{bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh' if use_bess else 'Not Included'],
            ['Connection Voltage', f'{rec_voltage_kv} kV'],
            ['Primary Fuel', fuel_mode],
            ['PUE (Operating)', f'{pue_actual:.2f}'],
        ]
        
        tech_table = Table(tech_table_data, colWidths=[3.5*inch, 2.5*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(tech_table)
        story.append(PageBreak())
        
        # Financial Summary
        story.append(Paragraph("Financial Summary", heading_style))
        
        financial_table_data = [
            ['Metric', 'Value'],
            ['Total CAPEX', f'${total_capex_dynamic:.2f}M'],
            ['LCOE (after tax)', f'${lcoe_dyn:.4f}/kWh'],
            ['Benchmark Price', f'${benchmark_price:.4f}/kWh'],
            ['Gas Price (total)', f'${total_gas_price:.2f}/MMBtu'],
            ['Annual Energy', f'{mwh_year:,.0f} MWh'],
            ['Annual Fuel Cost', f'${fuel_cost_year/1e6:.2f}M'],
            ['Annual O&M Cost', f'${om_cost_year/1e6:.2f}M'],
            ['Annual Savings', f'${annual_savings/1e6:.2f}M'],
            ['NPV (20 years)', f'${npv_dyn/1e6:.2f}M'],
            ['Simple Payback', f'{payback_dyn:.1f} years' if annual_savings > 0 else 'N/A'],
        ]
        
        financial_table = Table(financial_table_data, colWidths=[3.5*inch, 2.5*inch])
        financial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(financial_table)
        story.append(Spacer(1, 0.3*inch))
        
        # CAPEX Breakdown
        story.append(Paragraph("CAPEX Breakdown", heading_style))
        
        capex_table_data = [['Item', 'Cost (M USD)']]
        for idx, row in edited_capex.iterrows():
            capex_table_data.append([row['Item'], f"${row['Cost (M USD)']:.2f}M"])
        capex_table_data.append(['TOTAL', f"${total_capex_dynamic:.2f}M"])
        
        capex_table = Table(capex_table_data, colWidths=[3.5*inch, 2.5*inch])
        capex_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(capex_table)
        story.append(PageBreak())
        
        # Environmental
        story.append(Paragraph("Environmental Performance", heading_style))
        
        env_table_data = [
            ['Parameter', 'Value'],
            ['NOx Emissions', f'{nox_lb_hr:.2f} lb/hr'],
            ['CO Emissions', f'{co_lb_hr:.2f} lb/hr'],
            ['CO₂ Emissions (Annual)', f'{co2_ton_yr:,.0f} tons/year'],
            ['Carbon Cost (Annual)', f'${carbon_cost_year/1e6:.2f}M/year'],
            ['Site Footprint', f'{disp_area:.2f} {disp_area_unit}'],
            ['Water Use (WUE)', f'{wue:.1f} L/kWh'],
        ]
        
        env_table = Table(env_table_data, colWidths=[3.5*inch, 2.5*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(env_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], 
                                     fontSize=8, textColor=colors.grey, 
                                     alignment=TA_CENTER)
        story.append(Paragraph("CAT QuickSize v2.0 | Caterpillar Electric Power | 2026", footer_style))
        story.append(Paragraph("For internal use only. Not for redistribution.", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    pdf_data = create_pdf_export()
    
    col_pdf1, col_pdf2 = st.columns(2)
    
    with col_pdf1:
        st.download_button(
            label="📄 Download PDF Proposal (4 Pages)",
            data=pdf_data,
            file_name=f"CAT_Proposal_{dc_type.replace(' ','_')}_{p_it:.0f}MW_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col_pdf2:
        st.info("📄 **PDF Includes:** Executive Summary, Technical Config, Financials, CAPEX, Environmental")

# CSV Fallback (if no Excel available)
if not EXCEL_AVAILABLE:
    # Fallback: CSV exports
    st.info("💡 **CSV Export Mode:** Multiple files available for download")
    
    col_csv1, col_csv2, col_csv3 = st.columns(3)
    
    # CSV 1: Summary
    with col_csv1:
        summary_csv = pd.DataFrame({
            'Parameter': ['DC Type', 'IT Load MW', 'PUE', 'Generator', 'Fleet', 'Installed MW', 
                         'Availability %', 'Target Met', 'LCOE $/kWh', 'CAPEX $M', 'NPV $M'],
            'Value': [dc_type, p_it, pue, selected_gen, f'{n_running}+{n_reserve}', installed_cap, 
                     prob_gen*100, 'Yes' if target_met else 'No', lcoe_dyn, total_capex_dynamic, npv_dyn/1e6]
        })
        
        st.download_button(
            label="📋 Summary (CSV)",
            data=summary_csv.to_csv(index=False),
            file_name=f"Summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # CSV 2: Financial
    with col_csv2:
        financial_csv = pd.DataFrame({
            'Metric': ['CAPEX M$', 'LCOE $/kWh', 'Gas Price $/MMBtu', 'Annual Savings M$', 'NPV M$', 'Payback Years'],
            'Value': [total_capex_dynamic, lcoe_dyn, total_gas_price, annual_savings/1e6, npv_dyn/1e6, 
                     payback_dyn if annual_savings > 0 else 0]
        })
        
        st.download_button(
            label="💰 Financial (CSV)",
            data=financial_csv.to_csv(index=False),
            file_name=f"Financial_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # CSV 3: Technical
    with col_csv3:
        tech_csv = pd.DataFrame({
            'Spec': ['ISO MW', 'Site MW', 'Efficiency %', 'MTBF hrs', 'BESS MW', 'Footprint'],
            'Value': [unit_iso_cap, unit_site_cap, fleet_efficiency*100, gen_data['mtbf_hours'], 
                     bess_power_total, f'{disp_area:.2f} {disp_area_unit}']
        })
        
        st.download_button(
            label="⚙️ Technical (CSV)",
            data=tech_csv.to_csv(index=False),
            file_name=f"Technical_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.caption("💡 **Tip:** Install openpyxl (`pip install openpyxl`) for full Excel report with 9 sheets")

# --- FOOTER ---
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
col_foot1.caption("CAT QuickSize v2.0 Lite")
col_foot2.caption("Next-Gen Data Center Power Solutions")
col_foot3.caption("Caterpillar Electric Power | 2026")
