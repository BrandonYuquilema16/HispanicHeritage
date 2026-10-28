import io
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
st.set_page_config(page_title="Chile GDP Explorer | Hispanic Heritage Month", page_icon="🇨🇱", layout="wide")
START_YEAR = 1976
END_YEAR = 2026
SOURCE_URL = "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=CL"
API_URL = "https://api.worldbank.org/v2/country/CHL/indicator/NY.GDP.MKTP.CD"
st.markdown("""<style>.block-container{padding-top:1.5rem;max-width:1200px}h1,h2,h3{color:#15345a}div[data-testid='stMetric']{border:1px solid #e4e8ee;border-radius:12px;padding:12px}</style>""", unsafe_allow_html=True)
@st.cache_data(ttl=3600, show_spinner=False)
def load_gdp():
    response = requests.get(API_URL, params={"format":"json", "date":f"{START_YEAR}:{END_YEAR}", "per_page":100}, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("World Bank returned an unexpected response.")
    rows = []
    for item in payload[1]:
        if item.get("value") is not None:
            year = int(item["date"])
            if START_YEAR <= year <= END_YEAR:
                rows.append({"Year":year,"GDP (billion current US$)":float(item["value"])/1e9})
    if not rows:
        raise ValueError("No historical GDP observations were returned.")
    df = pd.DataFrame(rows).sort_values("Year").drop_duplicates("Year").reset_index(drop=True)
    df.insert(1,"x (years since 1976)",df["Year"]-START_YEAR)
    return df
def fit_model(df, degree):
    x = df["x (years since 1976)"].to_numpy(dtype=float)
    y = df["GDP (billion current US$)"].to_numpy(dtype=float)
    if len(df) <= degree:
        raise ValueError(f"A degree-{degree} polynomial requires at least {degree+1} distinct years.")
    # numpy Polynomial.fit scales the domain internally for numerical stability.
    fitted = np.polynomial.Polynomial.fit(x, y, deg=degree).convert()
    predicted = fitted(x)
    residuals = y-predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y-y.mean())**2))
    r2 = 1-sse/sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return fitted, predicted, residuals, r2, rmse
def equation(poly, label="G"):
    terms = []
    for power in range(len(poly.coef)-1, -1, -1):
        coef = float(poly.coef[power])
        if abs(coef) < 1e-12:
            continue
        variable = "" if power == 0 else "x" if power == 1 else f"x^{power}"
        magnitude = f"{abs(coef):.7g}"
        term = magnitude + (" " + variable if variable else "")
        if not terms:
            terms.append(("-" if coef < 0 else "") + term)
        else:
            terms.append((" - " if coef < 0 else " + ") + term)
    return f"{label}(x) = " + ("".join(terms) if terms else "0")
def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")
def pdf_report(raw, working, fitted_data, poly, r2, rmse, latest, prediction, degree, edited):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(8.5*inch,11*inch), rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [Paragraph("Chile GDP Explorer: 1976-2026", styles["Title"]), Spacer(1,10),
             Paragraph("Hispanic Heritage Month | Historical data and polynomial modeling",styles["Normal"]), Spacer(1,12),
             Paragraph("Source: World Bank, World Development Indicators, NY.GDP.MKTP.CD (GDP, current US$).",styles["Normal"]),
             Paragraph(f"Historical observations: {len(raw)}; latest available year: {latest}. GDP is measured in billions of current US dollars.",styles["Normal"]),
             Paragraph("2026 is a MODEL ESTIMATE, not an observed GDP value.",styles["Heading3"]),
             Paragraph(f"Degree: {degree}; R-squared: {r2:.5f}; RMSE: {rmse:.3f} billion USD.",styles["Normal"]),
             Paragraph(f"2026 model estimate: {prediction:.3f} billion USD.",styles["Normal"]),
             Paragraph(f"Working data edited: {'Yes' if edited else 'No'}; regression uses working data.",styles["Normal"]),Spacer(1,12),
             Paragraph("Polynomial equation (x = year - 1976)",styles["Heading2"]),
             Paragraph(equation(poly),styles["Normal"]),
             Paragraph("First derivative: "+equation(poly.deriv(),"G'"),styles["Normal"]),
             Paragraph("Second derivative: "+equation(poly.deriv(2),"G''"),styles["Normal"]),Spacer(1,10),
             Paragraph("Raw historical data (unmodified World Bank observations)",styles["Heading2"])]
    rows = [["Year","x","GDP (billion USD)"]]
    for _, row in raw.iterrows():
        rows.append([str(int(row["Year"])),str(int(row["x (years since 1976)"])),f'{row["GDP (billion current US$)"]:.3f}'])
    table = Table(rows, colWidths=[90,90,170],repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce9f5")),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3)]))
    story.extend([table,Spacer(1,12),Paragraph("Interpretation and limitations",styles["Heading2"]),
       Paragraph("The regression describes historical association, not causation. Current-US-dollar GDP is affected by prices and exchange rates. High-degree polynomials can produce unstable extrapolations; the 2026 prediction is not an official forecast. The dataset begins in 1976 for this project, not at Chile's founding.",styles["Normal"]),
       Paragraph("Context: Chile's economic history includes changing copper markets, political and economic reforms, the 1982 debt crisis, and the COVID-19 shock. The polynomial alone cannot explain these events.",styles["Normal"]),
       Paragraph("Data: "+SOURCE_URL,styles["Normal"])])
    doc.build(story)
    return buffer.getvalue
st.title("🇨🇱 Chile GDP Explorer")
st.caption("Hispanic Heritage Month • Historical GDP, polynomial regression, and calculus • 1976–2026")
st.info("GDP is annual economic output, not Chile's net worth. Historical observations come from the World Bank. Any 2026 estimate shown below is a mathematical extrapolation, not an official value.")
try:
    raw = load_gdp()
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    st.error(f"Could not retrieve verified historical data: {exc}")
    st.markdown(f"Check the [World Bank source]({SOURCE_URL}) and try refreshing. The app intentionally does not invent or substitute historical observations.")
    st.stop()
latest = int(raw["Year"].max())
if latest >= END_YEAR:
    st.warning("The source now includes 2026 data. The app still shows a 2026 model estimate separately and does not use 2026 observations to fit it.")
raw = raw.loc[raw["Year"] < END_YEAR].copy().reset_index(drop=True)
if len(raw) < 6:
    st.error("Insufficient historical observations for the required degree-3 or higher model.")
    st.stop()
st.sidebar.header("Model settings")
degree = st.sidebar.slider("Polynomial degree",min_value=3,max_value=min(6,len(raw)-1),value=3)
st.sidebar.caption("Degree 3 meets the assignment. Higher degrees may overfit.")
if st.sidebar.button("Restore original working data"):
    st.session_state["working_version"] = st.session_state.get("working_version",0)+1
    st.rerun()
version = st.session_state.get("working_version",0)
st.subheader("1. Historical source and raw data")
c1,c2,c3=st.columns(3)
c1.metric("First project year",str(int(raw["Year"].min())))
c2.metric("Latest observed year",str(latest))
c3.metric("Historical observations",str(len(raw)))
st.markdown(f"**Official source:** [World Bank — Chile GDP (current US$)]({SOURCE_URL}) | Indicator code: `NY.GDP.MKTP.CD` | Values converted from USD to **billions of current USD**.")
st.dataframe(raw,hide_index=True,use_container_width=True,height=270)
st.download_button("Download original historical data (CSV)",csv_bytes(raw),"chile_gdp_raw_1976_latest.csv","text/csv")
st.subheader("2. Editable working table")
st.write("Edit the GDP values below to test mathematical scenarios. The original historical table above is never changed. Edited values are **not** official historical observations.")
working_input = st.data_editor(raw[["Year","GDP (billion current US$)"]],hide_index=True,use_container_width=True,num_rows="fixed",disabled=["Year"],key=f"gdp_editor_{version}")
working = working_input.copy()
working["x (years since 1976)"] = working["Year"] - START_YEAR
working = working[["Year","x (years since 1976)","GDP (billion current US$)"]]
working["GDP (billion current US$)"] = pd.to_numeric(working["GDP (billion current US$)"],errors="coerce")
if working["GDP (billion current US$)"].isna().any() or not np.isfinite(working["GDP (billion current US$)"].to_numpy(dtype=float)).all():
    st.error("All working GDP values must be finite numbers.")
    st.stop()
if (working["GDP (billion current US$)"] < 0).any():
    st.error("GDP values cannot be negative in this working table.")
    st.stop()
edited = not np.allclose(working["GDP (billion current US$)"].to_numpy(),raw["GDP (billion current US$)"].to_numpy(),rtol=0,atol=1e-9)
if edited:
    st.warning("Scenario mode: the regression now uses edited data. Its results are NOT a fit to the unmodified World Bank series.")
st.download_button("Download working table (CSV)",csv_bytes(working),"chile_gdp_working.csv","text/csv")
try:
    poly, fitted, residuals, r2, rmse = fit_model(working,degree)
except (ValueError,np.linalg.LinAlgError) as exc:
    st.error(f"Model could not be fitted: {exc}")
    st.stop()
st.subheader("3. Polynomial equation and calculus")
st.latex(equation(poly).replace("x^","x^"))
st.caption("x = calendar year − 1976; G(x) is GDP in billions of current US dollars. Coefficients shown are rounded for display; calculations use full precision.")
st.markdown("**First derivative — estimated annual GDP change (billion USD/year)**")
st.latex(equation(poly.deriv(),"G'"))
st.markdown("**Second derivative — change in the annual GDP rate (billion USD/year²)**")
st.latex(equation(poly.deriv(2),"G''"))
with st.expander("Full-precision polynomial coefficients"):
    st.dataframe(pd.DataFrame({"Power of x":list(range(len(poly.coef))),"Coefficient":poly.coef}),hide_index=True)
st.subheader("4. Model quality")
a,b,c=st.columns(3)
a.metric("R²",f"{r2:.4f}" if np.isfinite(r2) else "Undefined")
b.metric("RMSE",f"${rmse:.2f}B")
c.metric("2026 estimate",f"${float(poly(END_YEAR-START_YEAR)):,.2f}B")
st.caption("R² measures in-sample fit; RMSE is the typical in-sample error scale. Neither guarantees accurate forecasts.")
st.subheader("5. Interactive GDP graph and slope explorer")
st.write("Move the **year slider** to move the highlighted dot along the regression curve. The tangent line shows the slope (instantaneous rate of change) at that year. Hover over any point to inspect its year and GDP.")
selected_graph_year = st.slider("Move the dot to a year", min_value=START_YEAR, max_value=END_YEAR, value=2026, step=1, key="graph_year")
selected_x = selected_graph_year - START_YEAR
selected_gdp = float(poly(selected_x))
selected_slope = float(poly.deriv()(selected_x))
selected_actual = raw.loc[raw["Year"] == selected_graph_year, "GDP (billion current US$)"]
g1, g2, g3 = st.columns(3)
g1.metric("Selected year", str(selected_graph_year))
g2.metric("Model GDP (not raw data)", f"${selected_gdp:,.2f}B")
g3.metric("Tangent slope", f"${selected_slope:+,.2f}B/year")
if not selected_actual.empty:
    st.caption(f"Official observed GDP for {selected_graph_year}: ${float(selected_actual.iloc[0]):,.2f}B. The model dot is a fitted value, which may differ from the observation.")
else:
    st.caption(f"No observed GDP in the selected dataset for {selected_graph_year}. This dot is a MODEL ESTIMATE, not an observation.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=raw["Year"],y=raw["GDP (billion current US$)"],mode="markers",name="Official historical GDP",marker=dict(color="#1768a5",size=7)))
if edited:
    fig.add_trace(go.Scatter(x=working["Year"],y=working["GDP (billion current US$)"],mode="markers",name="Edited working values",marker=dict(color="#e49b23",symbol="diamond",size=7)))
xx = np.linspace(START_YEAR,END_YEAR,400)
fig.add_trace(go.Scatter(x=xx,y=poly(xx-START_YEAR),mode="lines",name=f"Degree-{degree} polynomial",line=dict(color="#d83a4d",width=3)))
fig.add_trace(go.Scatter(x=[END_YEAR],y=[float(poly(END_YEAR-START_YEAR))],mode="markers",name="2026 MODEL ESTIMATE",marker=dict(color="#9a40ba",symbol="star",size=13)))
# Tangent at the slider-selected point: y = f(x0) + f'(x0)(x-x0).
tangent_years = np.linspace(max(START_YEAR, selected_graph_year-5), min(END_YEAR, selected_graph_year+5), 51)
tangent_values = selected_gdp + selected_slope*(tangent_years-selected_graph_year)
fig.add_trace(go.Scatter(x=tangent_years, y=tangent_values, mode="lines", name="Tangent / slope line", line=dict(color="#169c84", width=3, dash="dash")))
fig.add_trace(go.Scatter(x=[selected_graph_year], y=[selected_gdp], mode="markers", name="Movable year dot (slider)", marker=dict(color="#ffb000", size=16, line=dict(color="#222222", width=2)), hovertemplate="Year: %{x}<br>Model GDP: $%{y:,.2f}B<extra></extra>"))
fig.add_shape(type="line", x0=selected_graph_year, x1=selected_graph_year, y0=0, y1=selected_gdp, line=dict(color="#ffb000", dash="dot", width=1))
fig.add_shape(type="line", x0=START_YEAR, x1=selected_graph_year, y0=selected_gdp, y1=selected_gdp, line=dict(color="#ffb000", dash="dot", width=1))
fig.add_vline(x=latest,line_dash="dot",line_color="gray")
fig.update_layout(xaxis_title="Year",yaxis_title="GDP (billion current US$)",hovermode="x unified",legend=dict(orientation="h",y=1.13),margin=dict(t=60))
st.plotly_chart(fig,use_container_width=True)
st.caption("Yellow dot = selected model value; green dashed line = tangent (slope); blue dots = historical observations; red curve = polynomial regression. The slider moves the dot. Mouse hover displays precise values.")
st.download_button("Download graph (interactive HTML)",fig.to_html(include_plotlyjs=True).encode("utf-8"),"chile_gdp_graph.html","text/html")
resid_fig=go.Figure(go.Scatter(x=working["Year"],y=residuals,mode="markers+lines",marker=dict(color="#147c77"),name="Residual"))
resid_fig.add_hline(y=0,line_dash="dash")
resid_fig.update_layout(title="Residuals: actual working GDP − fitted GDP",xaxis_title="Year",yaxis_title="Residual (billion USD)")
st.plotly_chart(resid_fig,use_container_width=True)
st.subheader("6. Estimate one year or multiple years")
selected_year = st.number_input("Year to estimate",min_value=1976,max_value=2050,value=2026,step=1)
estimate=float(poly(int(selected_year)-START_YEAR))
kind="Extrapolation / MODEL ESTIMATE" if selected_year > latest else "Interpolation / fitted value"
st.metric(f"{int(selected_year)}: {kind}",f"${estimate:,.2f} billion")
st.write(f"First derivative at {int(selected_year)}: **{float(poly.deriv()(int(selected_year)-START_YEAR)):,.2f} billion USD/year**")
st.write(f"Second derivative at {int(selected_year)}: **{float(poly.deriv(2)(int(selected_year)-START_YEAR)):,.2f} billion USD/year²**")
if estimate < 0:
    st.error("The model predicts negative GDP, which is not economically meaningful. This demonstrates why polynomial extrapolation can fail.")
years_text=st.text_input("Multiple years (comma-separated)",value="2024, 2025, 2026")
   

years=[int(part.strip()) for part in years_text.split(",") if part.strip()]
    if not years or len(years)>30 or any(y<1976 or y>2050 for y in years):
        raise ValueError("Enter 1–30 years between 1976 and 2050.")
    predicted_table=pd.DataFrame({"Year":years,"Model GDP estimate (billion current US$)":[float(poly(y-START_YEAR)) for y in years],"Type":["EXTRAPOLATION — NOT OBSERVED" if y>latest else "FITTED VALUE — NOT RAW DATA" for y in years]})
    st.dataframe(predicted_table,hide_index=True,use_container_width=True)
    st.download_button("Download predictions (CSV)",csv_bytes(predicted_table),"chile_gdp_predictions.csv","text/csv")
except ValueError as exc:
    st.warning(f"Invalid multiple-year input: {exc}")
st.subheader("7. Download model and project report")
fit_table=working.copy()
fit_table["Fitted GDP (billion US$)"]=fitted
fit_table["Residual (billion US$)"]=residuals
st.download_button("Download model results and residuals (CSV)",csv_bytes(fit_table),"chile_gdp_model_results.csv","text/csv")
report=pdf_report(raw,working,fit_table,poly,r2,rmse,latest,float(poly(END_YEAR-START_YEAR)),degree,edited)
st.download_button("Download complete PDF report",report,"chile_gdp_project_report.pdf","application/pdf")
st.subheader("8. Historical and cultural interpretation")
st.markdown("""Chile is a Spanish-speaking South American country whose economy has been shaped by copper mining, trade, agriculture, services, and political and economic change. Historical events—including the 1982 debt crisis, the global financial crisis, and the COVID-19 pandemic—can affect economic indicators. **A polynomial curve alone cannot tell us why GDP changed.**
**Interpretation:** Compare the actual observations with the fitted curve, identify years with unusually large residuals, and investigate the historical context. Remember that GDP is not a measure of cultural value, income equality, or quality of life.
**Limitations:** The chosen series is *nominal GDP in current US dollars*, not inflation-adjusted GDP. Exchange rates and price changes affect the figures. Degree-3-or-higher polynomials may fit historical data while forecasting poorly. A 2026 extrapolation should never be labeled a measured value or official forecast. The project begins at 1976 by design, not at Chile's founding or the start of its economy.
""")
st.markdown(f"**Citation:** World Bank, World Development Indicators, *GDP (current US$), Chile*, indicator NY.GDP.MKTP.CD. [View source]({SOURCE_URL}). Accessed {datetime.now().date().isoformat()}.")
st.subheader("9. 🇨🇱 Chile fun fact")
fun_facts = [
    "Chile is one of the world's leading copper producers. Copper is important to its export economy, but GDP includes much more than mining.",
    "Chile stretches more than 4,000 kilometers from north to south, with climates ranging from the Atacama Desert to Patagonia.",
    "Chile's Atacama Desert is among the driest places on Earth, and its clear skies make it a major location for astronomical observatories.",
    "The Chilean peso is Chile's currency. GDP in current U.S. dollars can change when the peso-to-dollar exchange rate changes, even without an equal change in real output.",
    "Chile's national holiday celebrations, Fiestas Patrias, take place in September, the same month Hispanic Heritage Month begins in the United States."
]
fun_fact_index = st.selectbox("Choose a Chile fun fact", range(len(fun_facts)), format_func=lambda i: f"Fun fact {i+1}")
st.info(fun_facts[fun_fact_index])
st.caption("GDP measures annual economic production, NOT a country's net worth. Historical observations and mathematical predictions are clearly separated throughout this project.")
