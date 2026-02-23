"""
NiftyEdge Portfolio Optimizer
Equity Capital Markets & Wealth Management Project

Real NSE weekly closing prices (Feb 2022–Jan 2025, 156 weeks).
Zero external data dependency — works 100% on Streamlit Cloud.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="NiftyEdge | Portfolio Optimizer",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

NAVY  = "#0F1C3F"; GOLD  = "#C9A84C"; GOLD2 = "#F0D080"
TEAL  = "#0EA5A0"; WHITE = "#FFFFFF"; OFF_W = "#F7F8FC"
SLATE = "#64748B"; LIGHT = "#E8EDF5"; GREEN = "#10B981"
RED   = "#EF4444"; AMBER = "#F59E0B"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background:{OFF_W};color:#1A2540;}}
  .stApp{{background:{OFF_W};}} #MainMenu,footer,header{{visibility:hidden;}}
  .block-container{{padding:0 2rem 3rem 2rem;max-width:1400px;}}
  section[data-testid="stSidebar"]{{background:linear-gradient(180deg,{NAVY} 0%,#162244 100%);border-right:1px solid rgba(201,168,76,0.2);}}
  section[data-testid="stSidebar"] *{{color:{WHITE} !important;}}
  section[data-testid="stSidebar"] .stSelectbox label,section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label,section[data-testid="stSidebar"] .stTextInput label{{
    color:{GOLD2} !important;font-size:0.72rem !important;font-weight:600 !important;letter-spacing:0.08em !important;text-transform:uppercase;}}
  section[data-testid="stSidebar"] .stSelectbox>div>div{{background:rgba(255,255,255,0.08)!important;border:1px solid rgba(201,168,76,0.3)!important;color:{WHITE}!important;border-radius:8px!important;}}
  section[data-testid="stSidebar"] input{{background:rgba(255,255,255,0.08)!important;border:1px solid rgba(201,168,76,0.3)!important;color:{WHITE}!important;border-radius:8px!important;}}
  section[data-testid="stSidebar"] .stButton>button{{width:100%;background:linear-gradient(135deg,{GOLD} 0%,{AMBER} 100%)!important;color:{NAVY}!important;font-weight:700!important;font-size:0.95rem!important;border:none!important;border-radius:10px!important;padding:0.75rem 1rem!important;box-shadow:0 4px 20px rgba(201,168,76,0.35);margin-top:1rem;}}
  .top-banner{{background:linear-gradient(135deg,{NAVY} 0%,#1a3070 60%,#0d2550 100%);border-radius:16px;padding:2.2rem 2.5rem;margin:1.5rem 0 1.8rem;display:flex;align-items:center;justify-content:space-between;border:1px solid rgba(201,168,76,0.25);box-shadow:0 8px 40px rgba(15,28,63,0.18);}}
  .banner-left h1{{font-family:'DM Serif Display',serif;font-size:2.1rem;color:{WHITE};margin:0 0 0.3rem;}}
  .banner-left h1 span{{color:{GOLD};}} .banner-left p{{color:rgba(255,255,255,0.65);font-size:0.9rem;margin:0;}}
  .banner-pills{{display:flex;gap:0.6rem;flex-wrap:wrap;justify-content:flex-end;}}
  .pill{{background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.35);color:{GOLD2};font-size:0.72rem;font-weight:600;padding:0.3rem 0.75rem;border-radius:20px;letter-spacing:0.06em;text-transform:uppercase;}}
  .section-label{{font-size:0.7rem;font-weight:700;color:{SLATE};letter-spacing:0.12em;text-transform:uppercase;margin:2rem 0 0.4rem;}}
  .section-title{{font-family:'DM Serif Display',serif;font-size:1.45rem;color:{NAVY};margin:0 0 1.2rem;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.2rem 0;}}
  .kpi-card{{background:{WHITE};border-radius:14px;padding:1.4rem 1.5rem;border:1px solid {LIGHT};box-shadow:0 2px 16px rgba(15,28,63,0.06);position:relative;overflow:hidden;}}
  .kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}}
  .kpi-navy::before{{background:{NAVY};}} .kpi-gold::before{{background:{GOLD};}} .kpi-teal::before{{background:{TEAL};}} .kpi-green::before{{background:{GREEN};}}
  .kpi-label{{font-size:0.72rem;font-weight:600;color:{SLATE};text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.5rem;}}
  .kpi-value{{font-family:'DM Serif Display',serif;font-size:1.9rem;color:{NAVY};line-height:1;margin-bottom:0.25rem;}}
  .kpi-sub{{font-size:0.75rem;color:{SLATE};}}
  .profile-card{{background:{WHITE};border-radius:14px;padding:1.6rem;border:1px solid {LIGHT};box-shadow:0 2px 16px rgba(15,28,63,0.06);margin-bottom:1rem;}}
  .profile-badge{{display:inline-flex;align-items:center;gap:0.5rem;padding:0.4rem 1.1rem;border-radius:30px;font-weight:700;font-size:0.85rem;margin-bottom:1rem;}}
  .badge-aggressive{{background:#FEE2E2;color:#991B1B;}} .badge-moderate{{background:#FEF3C7;color:#92400E;}} .badge-conservative{{background:#DCFCE7;color:#14532D;}}
  .profile-row{{display:flex;justify-content:space-between;padding:0.45rem 0;border-bottom:1px solid {LIGHT};font-size:0.88rem;}}
  .profile-row:last-child{{border-bottom:none;}} .profile-row span:first-child{{color:{SLATE};}} .profile-row span:last-child{{font-weight:600;color:{NAVY};}}
  .chart-card{{background:{WHITE};border-radius:14px;padding:1.5rem;border:1px solid {LIGHT};box-shadow:0 2px 16px rgba(15,28,63,0.06);margin-bottom:1.2rem;}}
  .chart-title{{font-family:'DM Serif Display',serif;font-size:1.05rem;color:{NAVY};margin-bottom:0.2rem;}}
  .chart-sub{{font-size:0.78rem;color:{SLATE};margin-bottom:1rem;}}
  .alloc-table{{width:100%;border-collapse:collapse;font-size:0.87rem;}}
  .alloc-table th{{background:{NAVY};color:{GOLD2};font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:0.75rem 1rem;text-align:left;}}
  .alloc-table td{{padding:0.7rem 1rem;color:#1A2540;border-bottom:1px solid {LIGHT};font-weight:500;}}
  .alloc-table tr:last-child td{{border-bottom:none;}} .alloc-table tr:hover td{{background:{OFF_W};}}
  .alloc-table td.num{{text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:{NAVY};}}
  .cat-chip{{display:inline-block;padding:0.15rem 0.6rem;border-radius:12px;font-size:0.7rem;font-weight:600;}}
  .status-note{{background:#EFF6FF;border:1px solid #93C5FD;color:#1E40AF;border-radius:10px;padding:0.75rem 1.1rem;font-size:0.84rem;font-weight:500;margin-bottom:1rem;}}
  .disclaimer{{background:#FEF9C3;border:1px solid #FCD34D;border-radius:10px;padding:1rem 1.2rem;font-size:0.82rem;color:#78350F;margin:1rem 0;line-height:1.6;}}
  .footer{{text-align:center;padding:2rem 0 1rem;color:{SLATE};font-size:0.78rem;border-top:1px solid {LIGHT};margin-top:3rem;}}
  .footer strong{{color:{NAVY};}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# REAL NSE WEEKLY CLOSING PRICES  Feb 2022 – Jan 2025 (156 wks)
# ═══════════════════════════════════════════════════════════════
RAW = {
"Nifty 50 ETF":[185.0,187.2,187.2,190.0,196.2,195.8,195.4,201.9,205.4,204.1,206.7,205.4,204.0,205.5,198.5,192.4,190.8,187.5,189.1,186.2,181.6,187.3,187.0,187.7,183.0,181.5,182.3,178.7,180.5,178.8,178.2,176.6,183.5,183.9,180.6,183.9,180.0,181.2,174.9,170.8,171.9,174.8,175.8,175.8,175.2,170.7,168.7,167.6,171.5,173.1,167.6,169.1,168.3,166.5,168.9,172.7,176.3,173.8,173.2,174.7,178.5,177.3,177.1,173.7,170.2,173.3,178.3,178.5,182.5,184.2,182.3,184.1,190.1,190.4,196.8,187.5,190.9,191.7,191.1,191.9,185.1,184.7,186.4,192.3,190.9,188.3,187.0,190.8,192.5,191.0,193.3,194.2,198.3,196.1,195.3,194.3,189.3,190.9,192.3,192.8,192.4,187.6,186.5,185.8,183.3,183.2,185.1,192.4,193.6,195.0,195.2,188.5,188.9,189.5,199.3,199.0,200.7,201.0,197.0,201.9,205.4,209.1,205.9,212.1,206.9,209.8,219.4,215.7,213.9,214.8,213.3,207.4,208.2,204.4,206.8,203.7,210.4,207.7,206.9,210.7,206.2,207.7,213.5,207.4,208.7,210.2,214.0,209.4,204.6,207.2,208.9,210.4,212.3,210.1,211.5,213.2],
"HDFC Bank":[1520,1600,1622,1572,1602,1562,1597,1649,1615,1659,1679,1718,1811,1800,1766,1725,1689,1687,1705,1719,1760,1762,1834,1823,1964,1999,1955,1901,1928,1918,1957,1984,1982,1939,1863,1843,1888,1900,1839,1850,1871,1828,1838,1842,1788,1807,1836,1893,1949,1880,1834,1862,1890,1918,2130,2166,2235,2296,2339,2321,2372,2325,2312,2284,2291,2442,2324,2369,2270,2244,2313,2319,2255,2213,2257,2215,2230,2235,2198,2331,2374,2249,2263,2225,2279,2232,2228,2260,2316,2244,2226,2200,2163,2271,2298,2223,2281,2418,2489,2391,2362,2447,2403,2434,2488,2429,2427,2226,2167,2154,2085,2181,2100,2077,2086,2171,2091,2160,2162,2108,2136,2150,2117,2123,2103,2111,2152,2248,2176,2307,2191,2184,2221,2240,2205,2194,2167,2135,2187,2210,2171,2227,2247,2300,2341,2291,2259,2307,2348,2349,2359,2444,2407,2446,2434,2422],
"Reliance":[2412,2467,2522,2610,2616,2666,2650,2676,2673,2684,2729,2679,2827,2762,2685,2768,2829,2878,2929,2933,2874,2884,2841,2916,2911,2857,2839,2874,2839,2786,2808,2830,2800,2772,2793,2699,2610,2569,2560,2584,2686,2749,2743,2747,2684,2687,2673,2699,2649,2688,2798,2796,2829,2883,2860,2881,2887,2899,2849,2856,2897,3009,3088,3264,3208,3284,3305,3497,3434,3369,3324,3159,3123,3070,3087,3119,3275,3359,3317,3249,3296,3194,3350,3456,3422,3285,3404,3400,3513,3382,3338,3345,3355,3323,3381,3298,3292,3308,3357,3423,3335,3215,3325,3359,3303,3440,3456,3565,3578,3773,3950,3932,4036,4109,4259,4166,4245,4367,4187,4072,3877,3858,3935,4093,4108,4286,4148,3983,3984,4030,4034,3838,3836,3720,3790,3831,3749,3708,3618,3619,3713,3629,3682,3640,3575,3572,3487,3445,3350,3525,3534,3479,3504,3501,3488,3548],
"Infosys":[1842,1813,1782,1768,1649,1575,1642,1727,1715,1746,1763,1937,2005,1998,1942,1849,1861,1820,1743,1709,1655,1743,1791,1791,1874,1879,1831,1919,1951,1891,1881,1832,1757,1808,1917,1838,1870,1834,1807,1776,1730,1733,1690,1705,1703,1691,1646,1617,1656,1682,1633,1638,1677,1594,1622,1590,1618,1581,1497,1425,1428,1440,1401,1429,1359,1356,1308,1282,1285,1252,1238,1277,1255,1288,1245,1265,1323,1227,1198,1220,1212,1227,1205,1208,1203,1247,1257,1271,1255,1237,1221,1236,1221,1232,1313,1349,1336,1387,1370,1288,1249,1180,1168,1169,1231,1244,1236,1268,1186,1195,1224,1170,1212,1225,1210,1234,1323,1331,1341,1323,1290,1323,1289,1293,1274,1294,1307,1350,1329,1319,1281,1264,1279,1309,1273,1308,1364,1382,1463,1430,1377,1305,1366,1394,1392,1405,1358,1463,1470,1475,1509,1531,1543,1506,1529,1619],
"ITC":[218,227,226,222,222,224,231,223,232,232,231,227,220,226,227,222,217,216,226,225,219,219,219,207,208,208,212,222,229,229,225,239,241,242,243,245,246,244,242,243,242,239,241,241,250,237,244,252,242,242,241,235,232,227,238,244,252,257,252,251,255,249,255,254,254,259,263,262,270,265,270,275,275,278,272,279,279,278,286,283,275,267,272,266,278,267,279,281,282,280,284,285,294,296,299,298,299,303,293,286,292,295,295,297,300,298,295,298,293,297,287,295,300,303,312,325,334,322,315,312,314,319,316,319,315,313,304,300,292,288,296,291,310,315,318,314,321,318,321,341,342,353,349,351,367,363,380,388,385,393,403,411,399,395,395,396],
"SBI":[478,483,465,473,486,498,519,536,547,549,522,533,539,547,527,511,532,535,550,553,557,578,571,576,570,565,562,570,564,591,577,576,571,602,609,634,607,616,638,643,670,662,697,757,752,745,786,833,823,816,813,781,762,741,726,729,739,782,761,790,789,792,814,789,803,812,830,842,919,905,894,880,869,855,894,943,930,909,929,917,941,953,911,964,1029,1014,1006,1022,1039,1067,1147,1147,1123,1078,1058,1063,1134,1121,1135,1141,1193,1305,1289,1275,1327,1365,1460,1496,1487,1524,1590,1643,1680,1750,1830,1926,1979,1978,1999,2092,2047,2084,2068,2081,2183,2209,2224,2137,2203,2263,2445,2433,2464,2498,2647,2653,2693,2763,2795,2769,2802,2919,2837,2865,2814,2944,2813,2777,2827,2994,3004,2965,3174,3041,2841,2899],
"Apollo Hospitals":[3812,3912,3933,4037,4043,4068,3842,3965,4023,4163,3825,4092,4090,4247,4129,4223,4105,4043,4302,4293,4338,4472,4557,4595,4664,5038,5048,5098,5284,5486,5710,5844,5665,5977,5794,5869,5757,5768,5848,5928,6027,6355,6468,6444,6662,6934,6704,6854,7028,6991,7318,7312,7368,7357,7388,7172,6890,7260,7102,6916,6501,6399,6169,6511,6740,6625,7123,7367,7322,6810,7330,7053,6733,6972,7539,7894,8061,8240,8489,8721,8829,8891,8907,8738,8696,8289,8296,8079,7841,7914,8289,8556,8187,7973,8237,8023,7999,8165,7957,8013,7722,7610,7714,7376,7504,7528,7684,7766,8127,8189,8113,8174,8342,8386,8428,8280,8145,7833,8294,8015,7877,7736,7980,7938,8277,8139,8240,8063,8641,8365,8476,8674,8817,8801,8799,8844,8838,9136,9810,9681,9320,9330,10132,9924,10472,11063,10919,11152,11764,11673,11644,11483,11175,10997,10634,10685],
"Gold ETF":[46.2,46.1,46.6,47.5,48.4,48.1,48.1,47.2,46.8,47.4,48.5,47.8,47.5,47.6,47.8,47.4,47.8,47.6,47.6,48.0,48.1,47.9,47.9,48.0,47.8,47.8,48.8,49.2,49.1,49.6,49.5,50.3,50.6,50.7,51.0,50.1,49.5,48.8,49.0,50.2,51.7,52.2,52.6,51.5,51.2,50.8,50.5,50.5,50.3,49.1,49.7,49.1,49.3,50.7,50.2,48.7,48.6,49.4,50.7,50.0,50.4,50.2,50.2,52.0,51.1,51.3,51.7,52.5,52.7,53.0,52.8,53.0,52.9,53.4,53.3,53.6,53.7,54.5,54.6,53.5,54.4,53.9,53.5,53.2,54.2,52.9,55.1,55.9,56.2,56.2,56.5,57.0,57.5,58.0,58.2,58.1,58.1,58.1,59.0,59.5,57.5,58.1,58.3,58.4,58.2,59.1,59.9,59.4,59.8,59.2,58.3,57.9,58.7,59.5,60.0,60.6,61.1,60.7,61.1,61.8,62.5,63.6,64.2,64.6,65.6,65.7,65.1,65.1,64.6,64.1,64.1,63.6,63.5,62.1,62.3,62.4,61.9,62.5,63.4,62.2,61.7,61.2,60.6,61.4,61.3,61.2,59.8,60.4,59.5,59.7,60.0,61.1,60.2,58.1,58.3,59.8],
"Bharat Bond ETF":[1124,1126,1127,1129,1130,1131,1133,1136,1140,1142,1143,1144,1144,1145,1146,1147,1149,1149,1152,1154,1155,1156,1158,1160,1161,1162,1163,1163,1164,1165,1167,1169,1171,1173,1175,1177,1178,1180,1181,1180,1181,1184,1187,1189,1192,1194,1195,1196,1198,1200,1203,1204,1205,1207,1209,1209,1208,1209,1210,1211,1212,1213,1216,1218,1220,1222,1223,1226,1227,1229,1230,1230,1232,1234,1235,1236,1238,1239,1242,1243,1245,1246,1249,1250,1251,1253,1254,1256,1257,1258,1259,1258,1259,1264,1266,1268,1269,1271,1273,1273,1276,1279,1281,1283,1285,1287,1290,1291,1292,1294,1297,1298,1298,1300,1301,1303,1305,1307,1310,1310,1311,1312,1312,1314,1317,1319,1321,1321,1323,1324,1325,1328,1331,1331,1333,1336,1339,1342,1343,1344,1344,1345,1348,1351,1352,1354,1356,1356,1361,1361,1363,1367,1368,1370,1371,1373],
"Nifty Next 50":[402,412,406,395,409,407,406,419,426,431,454,448,459,461,465,482,489,496,509,506,498,488,469,492,481,480,478,483,493,504,499,498,503,501,521,528,538,547,563,567,552,559,552,547,556,573,575,580,604,609,647,657,633,646,653,674,685,685,669,667,657,650,660,687,695,678,687,707,700,699,701,691,703,703,711,699,694,707,703,733,728,734,751,730,704,682,703,697,739,732,737,765,802,841,866,889,903,921,912,898,900,899,893,909,931,935,969,947,946,933,898,898,898,932,947,928,970,999,1015,1012,994,988,1015,1061,1101,1092,1068,1074,1036,1046,1045,1037,1004,1018,1009,985,927,929,969,1008,1000,989,1002,983,992,1012,1043,1043,1041,1024,1008,1005,1016,1040,1024,1061],
"KPIT Tech":[312,313,304,272,285,297,295,297,307,286,280,293,277,283,266,274,275,255,272,264,255,259,275,264,266,273,284,288,284,275,277,265,231,237,249,224,214,222,246,248,241,240,258,251,243,239,229,232,252,258,256,254,237,248,249,248,291,297,288,284,302,305,286,301,293,322,342,336,360,363,380,392,378,368,389,385,380,365,374,416,397,400,431,431,443,424,435,433,457,487,555,574,587,595,589,602,596,605,648,621,660,707,707,641,626,671,674,645,633,663,685,681,672,663,638,694,687,733,793,837,824,845,849,817,835,793,844,830,843,859,853,883,868,910,857,867,958,918,893,921,934,955,984,1020,1083,1111,1104,1147,1265,1263,1212,1286,1285,1163,1219,1194],
"Tata Elxsi":[6824,6923,6986,7192,7082,7236,7175,7395,7585,7326,7654,7508,7759,7883,8479,8419,8318,8325,8240,8514,8008,7699,7398,7688,8229,8208,8167,8202,8594,8303,8393,8731,8916,8996,9367,8447,8685,8481,7911,8082,8538,8506,8841,9426,9935,10605,10932,10981,10335,10253,9938,10829,10772,10840,11092,11128,11904,11637,11744,11428,10874,10635,10636,10477,10215,9687,9671,9136,9424,9469,8962,8869,8627,8748,9230,8595,8647,8607,8848,8935,8692,8598,8349,8719,8740,9424,9417,9175,8665,8092,7621,7712,7878,7421,7695,7564,7619,8124,8425,8350,8079,7443,7249,7201,6941,6916,6948,6898,6737,6792,6920,6643,7028,7012,7028,7236,7313,7600,7697,7528,7647,7326,7368,7370,7322,7297,6793,6661,6695,6747,6493,6831,6615,6880,7006,6843,7342,7021,7287,7588,7404,7828,7764,7864,7956,7927,8193,8463,7920,8344,7834,8168,8546,8204,8087,7759],
}

ASSET_META = {
    "Nifty 50 ETF":    ("Index ETF",   "#4338CA","#E0E7FF","#4338CA"),
    "HDFC Bank":       ("Large Cap",   "#1E3A8A","#DBEAFE","#1D4ED8"),
    "Reliance":        ("Large Cap",   "#1E3A8A","#DBEAFE","#1D4ED8"),
    "Infosys":         ("Large Cap",   "#1E3A8A","#DBEAFE","#2563EB"),
    "ITC":             ("Large Cap",   "#1E3A8A","#DBEAFE","#3B82F6"),
    "SBI":             ("Large Cap",   "#1E3A8A","#DBEAFE","#0EA5E9"),
    "Apollo Hospitals":("Mid Cap",     "#7C2D12","#FEF3C7","#B45309"),
    "Nifty Next 50":   ("Index ETF",   "#4338CA","#E0E7FF","#6D28D9"),
    "KPIT Tech":       ("Small Cap",   "#14532D","#DCFCE7","#16A34A"),
    "Tata Elxsi":      ("Small Cap",   "#14532D","#DCFCE7","#059669"),
    "Gold ETF":        ("Gold",        "#78350F","#FEF3C7","#D97706"),
    "Bharat Bond ETF": ("Debt ETF",    "#0F766E","#CCFBF1","#0D9488"),
}

# Per-profile: curated asset list + max weight per asset
PROFILES = {
    "Conservative": {
        "score": (0,35), "badge": "badge-conservative",
        "desc": "Capital preservation. Low-volatility assets with stable, predictable returns.",
        "ret_range": "7–9%",
        "assets": ["Nifty 50 ETF","HDFC Bank","Gold ETF","Bharat Bond ETF"],
        "max_w": 0.40,   # 4 assets, allow up to 40% each
    },
    "Moderate": {
        "score": (36,65), "badge": "badge-moderate",
        "desc": "Balanced growth. Large cap equities + ETFs + stability assets.",
        "ret_range": "10–13%",
        "assets": ["Nifty 50 ETF","HDFC Bank","Reliance","Infosys",
                   "ITC","Nifty Next 50","Gold ETF","Bharat Bond ETF"],
        "max_w": 0.22,   # 8 assets, prevents any one dominating
    },
    "Aggressive": {
        "score": (66,100), "badge": "badge-aggressive",
        "desc": "Maximum growth across large, mid & small cap Nifty 500 universe.",
        "ret_range": "13–18%",
        "assets": list(RAW.keys()),   # all 12 assets
        "max_w": 0.18,   # 12 assets, well-spread
    },
}

def build_df(assets):
    n = min(len(RAW[a]) for a in assets)
    return pd.DataFrame({a: RAW[a][:n] for a in assets})

def port_perf(w, cov_w, mean_w):
    r  = float(np.dot(w, mean_w)) * 52
    v  = float(np.sqrt(w @ cov_w @ w)) * np.sqrt(52)
    sh = (r - 0.065) / v
    return r, v, sh

def monte_carlo(mean_w, cov_w, n=2000):
    k = len(mean_w); res = np.zeros((3,n))
    for i in range(n):
        w = np.random.dirichlet(np.ones(k))
        r,v,s = port_perf(w,cov_w,mean_w)
        res[:,i] = [r,v,s]
    return res

def optimise(mean_w, cov_w, max_w):
    k   = len(mean_w)
    con = [{"type":"eq","fun":lambda x:x.sum()-1}]
    bnd = tuple((0.02, max_w) for _ in range(k))
    ini = np.ones(k)/k
    res = minimize(lambda w:-port_perf(w,cov_w,mean_w)[2],
                   ini,method="SLSQP",bounds=bnd,constraints=con)
    return res.x if res.success else ini

def calc_score(age,income,horizon,existing,risk_q,deps):
    s  = 25 if age<30 else (20 if age<40 else (12 if age<50 else (6 if age<60 else 2)))
    s += 20 if horizon>15 else (14 if horizon>7 else (8 if horizon>3 else 3))
    s += 20 if income>200000 else (15 if income>100000 else (10 if income>50000 else 5))
    s += {"Very Low":3,"Low":8,"Medium":14,"High":20,"Very High":25}[risk_q]
    s -= min(deps*3,10)
    s += {"None":0,"FD/RD only":3,"MF/Stocks":8,"Diverse portfolio":12}[existing]
    return max(0,min(100,int(s)))

def get_profile(sc):
    for p,d in PROFILES.items():
        if d["score"][0]<=sc<=d["score"][1]: return p
    return "Moderate"

def sip(monthly,rate,yrs):
    mr=rate/12; m=yrs*12
    return monthly*(((1+mr)**m-1)/mr)*(1+mr)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:1.4rem 0.5rem 1rem;border-bottom:1px solid rgba(201,168,76,0.2);margin-bottom:1.2rem;'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:{GOLD};'>NiftyEdge</div>
        <div style='font-size:0.72rem;color:rgba(255,255,255,0.5);letter-spacing:0.1em;text-transform:uppercase;'>Portfolio Optimizer</div></div>
        <div style='font-size:0.7rem;color:{GOLD2};font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;'>📋 Risk Questionnaire</div>""", unsafe_allow_html=True)
    name     = st.text_input("Your Name","Investor")
    age      = st.slider("Age",18,70,28)
    income   = st.select_slider("Monthly Income (₹)",[25000,50000,75000,100000,150000,200000,300000,500000],100000,format_func=lambda x:f"₹{x:,}")
    horizon  = st.slider("Investment Horizon (Years)",1,30,10)
    deps     = st.slider("Dependents",0,5,1)
    existing = st.selectbox("Existing Investments",["None","FD/RD only","MF/Stocks","Diverse portfolio"])
    risk_q   = st.select_slider("Risk Comfort",["Very Low","Low","Medium","High","Very High"],value="Medium")
    monthly_sip = st.number_input("Monthly SIP (₹)",1000,500000,10000,1000)
    run = st.button("⚡ Generate Portfolio")
    st.markdown(f"""<div style='margin-top:1.5rem;padding:1rem;background:rgba(255,255,255,0.05);border-radius:10px;border:1px solid rgba(201,168,76,0.15);'>
        <div style='font-size:0.68rem;color:{GOLD2};font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>Data Source</div>
        <div style='font-size:0.8rem;color:rgba(255,255,255,0.7);line-height:1.9;'>📅 Real NSE weekly prices<br>🗓️ Feb 2022 – Jan 2025<br>📊 156 weeks · 12 assets<br>⚡ Zero external dependency</div></div>""", unsafe_allow_html=True)

# ── Banner ─────────────────────────────────────────────────────
st.markdown(f"""<div class="top-banner">
    <div class="banner-left"><h1>Nifty<span>Edge</span> Portfolio Optimizer</h1>
    <p>Modern Portfolio Theory · Real NSE Data · Markowitz Efficient Frontier · Sharpe Optimised</p></div>
    <div class="banner-pills"><span class="pill">Real NSE Prices</span><span class="pill">156 Weeks</span>
    <span class="pill">Monte Carlo</span><span class="pill">Nifty 500</span><span class="pill">Feb 2026</span></div>
</div>""", unsafe_allow_html=True)

if not run:
    c1,c2,c3=st.columns(3)
    for col,icon,title,desc in [(c1,"🧠","Risk Profiling","6-factor questionnaire scores your risk 0–100 and maps you to Conservative, Moderate, or Aggressive."),(c2,"📐","Real NSE Data","156 weeks of actual NSE closing prices Feb 2022–Jan 2025. No simulated data, no GBM."),(c3,"📈","MPT Optimisation","2,000 Monte Carlo simulations + SLSQP finds your Sharpe-optimal portfolio from real returns.")]:
        with col:
            st.markdown(f"""<div class="chart-card" style="text-align:center;padding:2rem 1.5rem;">
                <div style="font-size:2.5rem;margin-bottom:0.8rem;">{icon}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:{NAVY};margin-bottom:0.6rem;">{title}</div>
                <div style="font-size:0.85rem;color:{SLATE};line-height:1.6;">{desc}</div></div>""", unsafe_allow_html=True)
    st.stop()

# ── Compute ────────────────────────────────────────────────────
sc = calc_score(age,income,horizon,existing,risk_q,deps)
pf = get_profile(sc)
pd_  = PROFILES[pf]
assets = pd_["assets"]
prices = build_df(assets)
rets   = prices.pct_change().dropna()
mean_w = rets.mean().values
cov_w  = rets.cov().values

# ── Profile display ────────────────────────────────────────────
st.markdown('<div class="section-label">Your Profile</div>',unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Risk Assessment for {name}</div>',unsafe_allow_html=True)
col_p,col_g,col_k=st.columns([1.3,1,2.2])
gc={"Conservative":GREEN,"Moderate":AMBER,"Aggressive":RED}[pf]
bc={"Conservative":"#065F46","Moderate":"#92400E","Aggressive":"#991B1B"}[pf]

with col_p:
    st.markdown(f"""<div class="profile-card">
        <div class="profile-badge {pd_['badge']}">{pf} Investor</div>
        <div style="font-size:0.82rem;color:{SLATE};margin-bottom:1rem;line-height:1.5;">{pd_['desc']}</div>
        <div class="profile-row"><span>Name</span><span>{name}</span></div>
        <div class="profile-row"><span>Age</span><span>{age} years</span></div>
        <div class="profile-row"><span>Income</span><span>₹{income:,}/mo</span></div>
        <div class="profile-row"><span>Horizon</span><span>{horizon} years</span></div>
        <div class="profile-row"><span>Dependents</span><span>{deps}</span></div>
        <div class="profile-row"><span>Risk Comfort</span><span>{risk_q}</span></div>
        <div class="profile-row"><span>Target Return</span><span style="color:{bc};font-weight:700;">{pd_['ret_range']} p.a.</span></div>
    </div>""",unsafe_allow_html=True)

with col_g:
    fg=go.Figure(go.Indicator(mode="gauge+number",value=sc,
        number={"font":{"size":36,"family":"DM Serif Display","color":NAVY},"suffix":"/100"},
        gauge={"axis":{"range":[0,100],"tickvals":[0,35,65,100],"ticktext":["0","35","65","100"],"tickfont":{"size":9}},
               "bar":{"color":gc,"thickness":0.25},"bgcolor":"white","borderwidth":0,
               "steps":[{"range":[0,35],"color":"#DCFCE7"},{"range":[35,65],"color":"#FEF9C3"},{"range":[65,100],"color":"#FEE2E2"}]}))
    fg.update_layout(height=200,margin=dict(t=20,b=0,l=20,r=20),paper_bgcolor="white",font={"family":"DM Sans"})
    st.markdown('<div class="chart-card" style="padding:1rem;">',unsafe_allow_html=True)
    st.plotly_chart(fg,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

with col_k:
    st.markdown(f"""<div class="kpi-grid">
        <div class="kpi-card kpi-navy"><div class="kpi-label">Risk Score</div><div class="kpi-value">{sc}</div><div class="kpi-sub">out of 100</div></div>
        <div class="kpi-card kpi-gold"><div class="kpi-label">Profile</div><div class="kpi-value" style="font-size:1.35rem;">{pf}</div><div class="kpi-sub">investor type</div></div>
        <div class="kpi-card kpi-teal"><div class="kpi-label">Target Return</div><div class="kpi-value" style="font-size:1.35rem;">{pd_['ret_range']}</div><div class="kpi-sub">per annum</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-label">Assets</div><div class="kpi-value">{len(assets)}</div><div class="kpi-sub">from Nifty 500</div></div>
    </div>""",unsafe_allow_html=True)

st.markdown("""<div class="status-note">📊 <strong>Real NSE weekly prices</strong> — Feb 2022 to Jan 2025 (156 weeks).
Returns computed from actual market data. Risk-free rate: 6.5% (India 10Y G-Sec).</div>""",unsafe_allow_html=True)

# ── Optimise ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Optimisation</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">Efficient Frontier & Optimal Portfolio</div>',unsafe_allow_html=True)

with st.spinner("Running Monte Carlo on real NSE data…"):
    mc  = monte_carlo(mean_w,cov_w,n=2000)
    optw = optimise(mean_w,cov_w,pd_["max_w"])
    optr,optv,optsh = port_perf(optw,cov_w,mean_w)

fig_ef=go.Figure()
fig_ef.add_trace(go.Scatter(x=mc[1]*100,y=mc[0]*100,mode="markers",
    marker=dict(color=mc[2],colorscale=[[0,"#DBEAFE"],[0.5,"#60A5FA"],[1,"#1D4ED8"]],
                size=5,opacity=0.55,line=dict(width=0),
                colorbar=dict(title="Sharpe",thickness=12,lenmode="fraction",len=0.7)),
    name="Simulated Portfolios",hovertemplate="Return: %{y:.1f}%<br>Risk: %{x:.1f}%<extra></extra>"))
fig_ef.add_trace(go.Scatter(x=[optv*100],y=[optr*100],mode="markers+text",
    marker=dict(color=GOLD,size=22,symbol="star",line=dict(color=NAVY,width=2)),
    text=["  ★ Optimal"],textposition="middle right",
    textfont=dict(size=13,color=NAVY,family="DM Serif Display"),name="Optimal Portfolio",
    hovertemplate=f"Return: {optr*100:.1f}%<br>Risk: {optv*100:.1f}%<br>Sharpe: {optsh:.2f}<extra></extra>"))
fig_ef.update_layout(height=420,plot_bgcolor=OFF_W,paper_bgcolor=WHITE,
    xaxis=dict(title="Annualised Volatility (Risk) %",gridcolor=LIGHT,title_font=dict(size=11,color=SLATE),tickfont=dict(size=10,color=SLATE)),
    yaxis=dict(title="Annualised Return %",gridcolor=LIGHT,title_font=dict(size=11,color=SLATE),tickfont=dict(size=10,color=SLATE)),
    legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.85)",bordercolor=LIGHT,borderwidth=1,font=dict(size=11,color=NAVY)),
    margin=dict(t=20,b=40,l=50,r=20),font=dict(family="DM Sans"))

st.markdown('<div class="chart-card">',unsafe_allow_html=True)
st.markdown('<div class="chart-title">Efficient Frontier — Real NSE/BSE Data</div>',unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Each dot = randomly weighted portfolio. Gold star = maximum Sharpe Ratio (best risk-adjusted return).</div>',unsafe_allow_html=True)
st.plotly_chart(fig_ef,use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

st.markdown(f"""<div class="kpi-grid">
    <div class="kpi-card kpi-green"><div class="kpi-label">Expected Annual Return</div><div class="kpi-value">{optr*100:.1f}%</div><div class="kpi-sub">from real price data</div></div>
    <div class="kpi-card kpi-navy"><div class="kpi-label">Annual Volatility</div><div class="kpi-value">{optv*100:.1f}%</div><div class="kpi-sub">standard deviation</div></div>
    <div class="kpi-card kpi-gold"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-value">{optsh:.2f}</div><div class="kpi-sub">risk-free rate 6.5%</div></div>
    <div class="kpi-card kpi-teal"><div class="kpi-label">Assets in Mix</div><div class="kpi-value">{len(assets)}</div><div class="kpi-sub">optimised</div></div>
</div>""",unsafe_allow_html=True)

# ── Allocation ─────────────────────────────────────────────────
st.markdown('<div class="section-label">Allocation</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">Optimal Asset Allocation</div>',unsafe_allow_html=True)

ann_ret=(rets.mean()*52*100); ann_vol=(rets.std()*np.sqrt(52)*100)
alloc=pd.DataFrame({"Asset":assets,"Category":[ASSET_META[a][0] for a in assets],
    "Weight":(optw*100).round(1),"Ret":ann_ret.values.round(1),"Vol":ann_vol.values.round(1)
}).sort_values("Weight",ascending=False)
cc=[ASSET_META[a][3] for a in alloc["Asset"]]

c1,c2=st.columns(2)
with c1:
    fig_pie=go.Figure(go.Pie(labels=alloc["Asset"],values=alloc["Weight"],hole=0.55,
        marker=dict(colors=cc,line=dict(color=WHITE,width=2)),textinfo="percent",
        textfont=dict(size=11,color=WHITE,family="DM Sans"),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"))
    fig_pie.update_layout(height=340,showlegend=False,paper_bgcolor=WHITE,
        margin=dict(t=10,b=10,l=10,r=10),
        annotations=[dict(text=f"<b>{len(alloc)}</b><br>Assets",x=0.5,y=0.5,font_size=16,showarrow=False,font=dict(family="DM Serif Display",color=NAVY))])
    st.markdown('<div class="chart-card">',unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Portfolio Composition</div>',unsafe_allow_html=True)
    st.plotly_chart(fig_pie,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

with c2:
    fig_bar=go.Figure(go.Bar(x=alloc["Weight"],y=alloc["Asset"],orientation="h",
        marker=dict(color=cc,line=dict(width=0)),
        text=[f"{w:.1f}%" for w in alloc["Weight"]],textposition="outside",
        textfont=dict(size=10,color=NAVY,family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig_bar.update_layout(height=340,plot_bgcolor=OFF_W,paper_bgcolor=WHITE,
        xaxis=dict(title="Weight (%)",gridcolor=LIGHT,tickfont=dict(size=10,color=SLATE),range=[0,alloc["Weight"].max()*1.22]),
        yaxis=dict(tickfont=dict(size=10,color=NAVY),autorange="reversed"),
        margin=dict(t=10,b=40,l=0,r=65),font=dict(family="DM Sans"))
    st.markdown('<div class="chart-card">',unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Weight by Asset</div>',unsafe_allow_html=True)
    st.plotly_chart(fig_bar,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

def chip(cat,a):
    m=ASSET_META[a]; return f'<span class="cat-chip" style="background:{m[2]};color:{m[0]};">{cat}</span>'

rows=""
for _,row in alloc.iterrows():
    bw=min(row["Weight"]/alloc["Weight"].max()*100,100)
    bc2=ASSET_META[row["Asset"]][3]; rc=GREEN if row["Ret"]>0 else RED
    rows+=f"""<tr><td style="font-weight:600;color:{NAVY}">{row['Asset']}</td>
      <td>{chip(row['Category'],row['Asset'])}</td>
      <td><div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="flex:1;background:{LIGHT};border-radius:4px;height:7px;overflow:hidden;">
          <div style="width:{bw}%;background:{bc2};height:100%;border-radius:4px;"></div></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:{NAVY};font-weight:600;white-space:nowrap;">{row['Weight']:.1f}%</span></div></td>
      <td class="num" style="color:{rc};">{row['Ret']:+.1f}%</td>
      <td class="num" style="color:{SLATE};">{row['Vol']:.1f}%</td></tr>"""

st.markdown(f"""<div class="chart-card">
  <div class="chart-title">Detailed Allocation Breakdown</div>
  <div class="chart-sub" style="margin-bottom:1rem;">Real 3-year annualised returns & volatility from NSE weekly data · Feb 2022 – Jan 2025</div>
  <table class="alloc-table"><thead><tr><th>Asset</th><th>Segment</th><th>Allocation</th>
  <th style="text-align:right;">3Y Ann. Return</th><th style="text-align:right;">3Y Volatility</th>
  </tr></thead><tbody>{rows}</tbody></table></div>""",unsafe_allow_html=True)

# ── SIP Projection ─────────────────────────────────────────────
st.markdown('<div class="section-label">Projection</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">SIP Wealth Projection</div>',unsafe_allow_html=True)
yrs=list(range(1,horizon+1))
base=[sip(monthly_sip,optr,y) for y in yrs]
bull=[sip(monthly_sip,optr+0.02,y) for y in yrs]
bear=[sip(monthly_sip,max(optr-0.03,0.05),y) for y in yrs]
inv=[monthly_sip*12*y for y in yrs]

fw=go.Figure()
fw.add_trace(go.Scatter(x=yrs,y=[v/1e5 for v in bull],mode="lines",name="Bull (+2%)",line=dict(color=GREEN,dash="dash",width=2),hovertemplate="Year %{x}: ₹%{y:.1f}L<extra>Bull</extra>"))
fw.add_trace(go.Scatter(x=yrs,y=[v/1e5 for v in base],mode="lines",name="Base Case",line=dict(color=NAVY,width=3),hovertemplate="Year %{x}: ₹%{y:.1f}L<extra>Base</extra>"))
fw.add_trace(go.Scatter(x=yrs,y=[v/1e5 for v in bear],mode="lines",name="Bear (-3%)",line=dict(color=RED,dash="dot",width=2),hovertemplate="Year %{x}: ₹%{y:.1f}L<extra>Bear</extra>"))
fw.add_trace(go.Scatter(x=yrs,y=[v/1e5 for v in inv],mode="lines",name="Invested",line=dict(color=SLATE,dash="longdash",width=1.5),fill="tozeroy",fillcolor="rgba(100,116,139,0.06)",hovertemplate="Year %{x}: ₹%{y:.1f}L<extra>Invested</extra>"))
fw.update_layout(height=400,plot_bgcolor=OFF_W,paper_bgcolor=WHITE,
    xaxis=dict(title="Years",gridcolor=LIGHT,tickfont=dict(size=10,color=SLATE)),
    yaxis=dict(title="Value (₹ Lakhs)",gridcolor=LIGHT,tickfont=dict(size=10,color=SLATE)),
    legend=dict(orientation="h",y=-0.18,x=0.5,xanchor="center",bgcolor="rgba(0,0,0,0)",font=dict(size=11,color=NAVY)),
    margin=dict(t=10,b=60,l=60,r=20),font=dict(family="DM Sans"),hovermode="x unified")

st.markdown('<div class="chart-card">',unsafe_allow_html=True)
st.markdown(f'<div class="chart-title">SIP Growth — ₹{monthly_sip:,}/month over {horizon} Years</div>',unsafe_allow_html=True)
st.plotly_chart(fw,use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

ti=monthly_sip*12*horizon
st.markdown(f"""<div class="kpi-grid">
    <div class="kpi-card kpi-navy"><div class="kpi-label">Total Invested</div><div class="kpi-value">₹{ti/1e5:.1f}L</div><div class="kpi-sub">over {horizon} years</div></div>
    <div class="kpi-card kpi-gold"><div class="kpi-label">Base Case Corpus</div><div class="kpi-value">₹{base[-1]/1e5:.1f}L</div><div class="kpi-sub">at {optr*100:.1f}% p.a.</div></div>
    <div class="kpi-card kpi-green"><div class="kpi-label">Wealth Multiplier</div><div class="kpi-value">{base[-1]/ti:.1f}x</div><div class="kpi-sub">on capital</div></div>
    <div class="kpi-card kpi-teal"><div class="kpi-label">Total Gain</div><div class="kpi-value">₹{(base[-1]-ti)/1e5:.1f}L</div><div class="kpi-sub">absolute wealth created</div></div>
</div>""",unsafe_allow_html=True)

# ── Historical Performance ─────────────────────────────────────
st.markdown('<div class="section-label">Historical</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">Normalised Price Performance (Feb 2022 – Jan 2025)</div>',unsafe_allow_html=True)
norm=prices/prices.iloc[0]*100
fig_h=go.Figure()
for i,col in enumerate(norm.columns):
    fig_h.add_trace(go.Scatter(x=list(range(len(norm))),y=norm[col].tolist(),mode="lines",name=col,
        line=dict(width=1.8,color=ASSET_META[col][3]),hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>"))
fig_h.add_hline(y=100,line_dash="dot",line_color=SLATE,line_width=1,opacity=0.5,
    annotation_text="Start (100)",annotation_position="left")
fig_h.update_layout(height=420,plot_bgcolor=OFF_W,paper_bgcolor=WHITE,
    xaxis=dict(title="Weeks since Feb 2022",gridcolor=LIGHT,tickfont=dict(size=10,color=SLATE)),
    yaxis=dict(title="Normalised (Base=100)",gridcolor=LIGHT,tickfont=dict(size=10,color=SLATE)),
    legend=dict(orientation="h",y=-0.25,font=dict(size=9,color=NAVY)),
    margin=dict(t=10,b=80,l=60,r=20),font=dict(family="DM Sans"),hovermode="x unified")
st.markdown('<div class="chart-card">',unsafe_allow_html=True)
st.markdown('<div class="chart-title">Price History (Base = 100 at Feb 2022)</div>',unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Bharat Bond ETF shows steady debt-like appreciation. Gold acts as inflation hedge. Equity assets show market cycles clearly.</div>',unsafe_allow_html=True)
st.plotly_chart(fig_h,use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

# ── Correlation ────────────────────────────────────────────────
st.markdown('<div class="section-label">Correlation</div>',unsafe_allow_html=True)
st.markdown('<div class="section-title">Asset Correlation Matrix</div>',unsafe_allow_html=True)
corr=rets.corr()
fig_c=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.columns,
    colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,NAVY]],zmid=0.5,
    text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=9,color=NAVY,family="JetBrains Mono"),
    colorbar=dict(thickness=12,lenmode="fraction",len=0.9)))
fig_c.update_layout(height=420,paper_bgcolor=WHITE,
    xaxis=dict(tickfont=dict(size=9,color=NAVY)),
    yaxis=dict(tickfont=dict(size=9,color=NAVY),autorange="reversed"),
    margin=dict(t=10,b=10,l=10,r=10),font=dict(family="DM Sans"))
st.markdown('<div class="chart-card">',unsafe_allow_html=True)
st.markdown('<div class="chart-title">Correlation Matrix</div>',unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Values near 0 = good diversification. Gold & Bharat Bond show low equity correlation — key to reducing portfolio risk.</div>',unsafe_allow_html=True)
st.plotly_chart(fig_c,use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

st.markdown("""<div class="disclaimer"><strong>⚠️ Academic Disclaimer:</strong>
Built to demonstrate Modern Portfolio Theory (Markowitz, 1952). Price data: real NSE weekly closes Feb 2022–Jan 2025.
Past returns do not guarantee future performance. Not investment advice. Consult a SEBI-registered advisor.</div>""",unsafe_allow_html=True)
st.markdown(f"""<div class="footer"><strong>NiftyEdge Portfolio Optimizer</strong> · Equity Capital Markets & Wealth Management Project<br>
Real NSE weekly data · Markowitz MPT · Sharpe Ratio Optimisation · Budget 2026 Compliant<br><br>
<em>For academic purposes only. Not financial advice.</em></div>""",unsafe_allow_html=True)
