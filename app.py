"""
App Streamlit: precipitação ERA5-Land num ponto escolhido no mapa.

Duas fontes:
  - Open-Meteo (sem chave; uso não comercial)
  - CDS / Copernicus (cdsapi; requer ~/.cdsapirc e licença do ERA5-Land aceite)

Executar:  streamlit run app.py
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="ERA5-Land · Precipitação", layout="wide")
st.title("Precipitação ERA5-Land num ponto")

# ---------------------------------------------------------------- estado
st.session_state.setdefault("pt", (39.5, -8.0))
st.session_state.setdefault("df", None)

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    fonte = st.radio("Fonte", ["Open-Meteo (rápido, sem chave)", "CDS Copernicus (oficial)"])
    lat = st.number_input("Latitude", -90.0, 90.0, st.session_state.pt[0], format="%.4f")
    lon = st.number_input("Longitude", -180.0, 180.0, st.session_state.pt[1], format="%.4f")
    st.session_state.pt = (lat, lon)
    hoje = date.today()
    d0 = st.date_input("Data inicial", hoje - timedelta(days=36), min_value=date(1950, 1, 1), max_value=hoje)
    d1 = st.date_input("Data final", hoje - timedelta(days=6), min_value=date(1950, 1, 1), max_value=hoje)
    res = st.selectbox("Resolução temporal", ["Diária (mm/dia)", "Horária (mm/h)"])
    mostrar_malha = st.checkbox("Mostrar malha ERA5-Land (0,1°)", value=True)
    n_cel = st.slider("Extensão da malha (células em redor do ponto)", 3, 15, 8, disabled=not mostrar_malha)
    go = st.button("Obter dados", type="primary", use_container_width=True)
    st.caption("ERA5-Land: grelha de 0,1° (~9 km), desde 1950, ~5 dias de atraso.")

# ---------------------------------------------------------------- funções
def via_open_meteo(lat, lon, d0, d1, diaria):
    var = "daily=precipitation_sum" if diaria else "hourly=precipitation"
    st.session_state.aviso = None
    for modelo in ("era5_land", "era5"):
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}"
            f"&{var}&models={modelo}&timezone=UTC"
        )
        r = requests.get(url, timeout=120)
        j = r.json()
        if r.status_code != 200 or j.get("error"):
            raise RuntimeError(j.get("reason", f"HTTP {r.status_code}"))
        bloco = j["daily" if diaria else "hourly"]
        chaves = [k for k in bloco if k != "time"]
        df = pd.DataFrame({"time_utc": pd.to_datetime(bloco["time"])})
        df["precip_mm"] = pd.to_numeric(pd.Series(bloco[chaves[0]]), errors="coerce")
        if df.precip_mm.notna().any():
            if modelo != "era5_land":
                st.session_state.aviso = (
                    "O ERA5-Land não devolveu valores para este ponto/período "
                    "(possível célula de mar ou datas demasiado recentes). "
                    "Mostrados dados ERA5 (0,25°)."
                )
            if df.precip_mm.isna().any():
                st.session_state.aviso = (st.session_state.aviso or "") + (
                    f" Há {int(df.precip_mm.isna().sum())} registos sem valor (normalmente os últimos dias)."
                )
            return df, (j["latitude"], j["longitude"])
    raise RuntimeError("Sem dados para este ponto/período. Experimente um ponto em terra e datas com mais de ~7 dias.")


def via_cds(lat, lon, d0, d1, diaria):
    import cdsapi
    import xarray as xr

    # um dia antes e um depois: necessário para desacumular corretamente
    a, b = d0 - timedelta(days=1), d1 + timedelta(days=1)
    dias = pd.date_range(a, b)
    pedido = {
        "variable": ["total_precipitation"],
        "year": sorted({f"{d.year}" for d in dias}),
        "month": sorted({f"{d.month:02d}" for d in dias}),
        "day": sorted({f"{d.day:02d}" for d in dias}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1],  # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    with tempfile.TemporaryDirectory() as tmp:
        alvo = Path(tmp) / "era5land.nc"
        cdsapi.Client(quiet=True).retrieve("reanalysis-era5-land", pedido).download(str(alvo))
        ds = xr.open_dataset(alvo)
        tcoord = "valid_time" if "valid_time" in ds.coords else "time"
        tp = ds["tp"]
        if "expver" in tp.dims:  # dados recentes (ERA5T) vêm com duas versões, uma delas toda NaN
            tp = tp.sel(expver=tp["expver"].values[0]).combine_first(tp.sel(expver=tp["expver"].values[-1]))
        # célula válida (com terra) mais próxima do ponto; o mar é NaN no ERA5-Land
        valida = tp.notnull().any(tcoord if tcoord in tp.dims else "time")
        lats, lons = tp.latitude.values, tp.longitude.values
        melhor = None
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                if bool(valida.isel(latitude=i, longitude=j)):
                    d = (la - lat) ** 2 + (lo - lon) ** 2
                    if melhor is None or d < melhor[0]:
                        melhor = (d, la, lo)
        if melhor is None:
            raise RuntimeError("Sem dados ERA5-Land perto deste ponto (mar?). Escolha um ponto em terra.")
        pt = tp.sel(latitude=melhor[1], longitude=melhor[2])
        glat, glon = float(melhor[1]), float(melhor[2])
        s = pt.to_series()
        s.index = pd.to_datetime(pt[tcoord].values)
        ds.close()

    # ERA5-Land acumula desde as 00 UTC; a hora 01 já é o acumulado de 1 h
    h = s.diff()
    h[h.index.hour == 1] = s[s.index.hour == 1]
    h = (h * 1000).clip(lower=0)  # m -> mm
    h = h.dropna()
    # cada valor refere-se à hora que termina no instante indicado
    h.index = h.index - pd.Timedelta(hours=1)
    h = h[(h.index >= pd.Timestamp(d0)) & (h.index < pd.Timestamp(d1) + pd.Timedelta(days=1))]
    if diaria:
        h = h.resample("D").sum()
    return h.rename("precip_mm").rename_axis("time_utc").reset_index(), (glat, glon)


# ---------------------------------------------------------------- mapa
col_mapa, col_res = st.columns([3, 2])
with col_mapa:
    st.caption("Clique no mapa para escolher o ponto. Laranja: célula do ponto · Verde: célula dos dados descarregados.")
    m = folium.Map(location=st.session_state.pt, zoom_start=6 if st.session_state.pt == (39.5, -8.0) else 10)
    PASSO = 0.1  # resolução do ERA5-Land; os centros das células estão em múltiplos de 0,1°
    if mostrar_malha:
        clat, clon = round(st.session_state.pt[0] / PASSO) * PASSO, round(st.session_state.pt[1] / PASSO) * PASSO
        malha = folium.FeatureGroup(name="Malha ERA5-Land")
        for i in range(-n_cel, n_cel + 1):
            for j in range(-n_cel, n_cel + 1):
                la, lo = clat + i * PASSO, clon + j * PASSO
                folium.Rectangle(
                    [(la - PASSO / 2, lo - PASSO / 2), (la + PASSO / 2, lo + PASSO / 2)],
                    color="#0b6e8f", weight=1, opacity=0.6, fill=True, fill_opacity=0.04,
                ).add_to(malha)
        malha.add_to(m)
        # célula que contém o ponto escolhido
        folium.Rectangle(
            [(clat - PASSO / 2, clon - PASSO / 2), (clat + PASSO / 2, clon + PASSO / 2)],
            color="#e67e22", weight=3, fill=True, fill_opacity=0.25,
            tooltip=f"Célula do ponto (centro {clat:.1f}, {clon:.1f})",
        ).add_to(m)
        # célula efetivamente usada nos dados descarregados (pode diferir se o ponto estava no mar)
        if st.session_state.df is not None:
            gl, go_ = st.session_state.df[1]
            folium.Rectangle(
                [(gl - PASSO / 2, go_ - PASSO / 2), (gl + PASSO / 2, go_ + PASSO / 2)],
                color="#27ae60", weight=3, fill=True, fill_opacity=0.25,
                tooltip=f"Célula dos dados (centro {gl:.2f}, {go_:.2f})",
            ).add_to(m)
    folium.Marker(st.session_state.pt).add_to(m)
    out = st_folium(m, height=480, use_container_width=True, returned_objects=["last_clicked"])
    if out and out.get("last_clicked"):
        novo = (round(out["last_clicked"]["lat"], 4), round(out["last_clicked"]["lng"], 4))
        if novo != st.session_state.pt:
            st.session_state.pt = novo
            st.rerun()

# ---------------------------------------------------------------- pedido
if go:
    if d0 > d1:
        st.error("A data inicial tem de ser anterior ou igual à final.")
    else:
        diaria = res.startswith("Diária")
        try:
            with st.spinner("A obter dados…" + (" (a fila do CDS pode demorar minutos)" if "CDS" in fonte else "")):
                fn = via_open_meteo if fonte.startswith("Open") else via_cds
                df, celula = fn(*st.session_state.pt, d0, d1, diaria)
            st.session_state.df = (df, celula, diaria, d0, d1)
        except Exception as e:
            st.error(f"Falhou: {e}")

# ---------------------------------------------------------------- resultados
with col_res:
    if st.session_state.df is not None:
        df, celula, diaria, a, b = st.session_state.df
        c1, c2 = st.columns(2)
        c1.metric("Total no período (mm)", f"{df.precip_mm.sum():.1f}")
        c2.metric("Máximo diário (mm)" if diaria else "Máximo horário (mm)", f"{df.precip_mm.max():.1f}")
        st.caption(f"{len(df)} registos · célula ERA5-Land: {celula[0]:.2f}, {celula[1]:.2f}")
        if st.session_state.get("aviso"):
            st.warning(st.session_state.aviso)
        st.bar_chart(df.set_index("time_utc")["precip_mm"])
        nome = f"era5land_prec_{'diaria' if diaria else 'horaria'}_{celula[0]:.2f}_{celula[1]:.2f}_{a}_{b}.csv"
        st.download_button("Descarregar CSV", df.to_csv(index=False).encode(), nome, "text/csv", use_container_width=True)
        with st.expander("Ver tabela"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Escolha o ponto e o período e clique em «Obter dados».")
