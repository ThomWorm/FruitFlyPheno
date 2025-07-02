import panel as pn
import holoviews as hv
import geoviews as gv

# Add custom CSS for dark side panel
pn.extension(raw_css=[
    """
    .fflies-dark-card {
        background: #23272b !important;
        color: #fff !important;
        border-radius: 10px;
        border: 1px solid #444;
    }
    .fflies-dark-card .bk-panel-models-markdown {
        color: #fff !important;
    }
    .fflies-dark-card .bk-input, .fflies-dark-card .bk-slider-title {
        color: #fff !important;
    }
    .fflies-dark-card .bk-slider-horizontal .bk-slider-bar {
        background: #444 !important;
    }
    """
])

gv.extension("bokeh")
hv.extension("bokeh")

def create_fflies_dashboard(da, species, detection_date, latitude, longitude, generations, save_path=None):
    """
    Create the interactive dashboard layout for FfliesOutput.plot.
    """
    precomputed = {
        "Likely Completion Date": da.mean(dim="year"),
        "Range of Likely Completion Dates": da.max(dim="year") - da.min(dim="year"),
    }
    da = da.sortby(["latitude", "longitude"])
    years = da.coords["year"].values
    year_labels = {f"sim{y}": y for y in da.coords["year"].values}
    custom_layers = {"Mean (All Years)": "mean", "Range (All Years)": "range"}
    year_options = {**custom_layers, **year_labels}
    select_styles = {"color": "#000", "font-size": "16px", "font-family": "Inter, Arial, sans-serif", "background": "#fff"}
    year_select = pn.widgets.Select(
        name="Year / Layer",
        options=year_options,
        value="mean",
        styles=select_styles,
        width=150,
    )
    gen_select = pn.widgets.Select(
        name="Generation",
        options=generations.tolist(),
        value=generations[2] if len(generations) > 2 else generations[0],
        styles=select_styles,
        width=150,
    )
    alpha_slider = pn.widgets.FloatSlider(
        name="Transparency",
        start=0.0,
        end=1.0,
        step=0.05,
        value=0.8,
        styles=select_styles,
        width=150,
    )
    clim_per_gen = {}
    clim_range_per_gen = {}
    for gen in generations:
        gen_data = da.sel(generation=gen)
        clim_per_gen[gen.item() if hasattr(gen, "item") else gen] = (
            float(gen_data.min()),
            float(gen_data.max()),
        )
        range_data = precomputed["Range of Likely Completion Dates"].sel(
            generation=gen
        )
        clim_range_per_gen[gen.item() if hasattr(gen, "item") else gen] = (
            float(range_data.min()),
            float(range_data.max()),
        )
    def make_plot(year_or_stat, generation, alpha):
        gen_key = generation.item() if hasattr(generation, "item") else generation
        if year_or_stat == "mean":
            sliced = precomputed["Likely Completion Date"].sel(
                generation=generation
            )
            clim = clim_per_gen[gen_key]
            cmap = "Viridis"
        elif year_or_stat == "range":
            sliced = precomputed["Range of Likely Completion Dates"].sel(
                generation=generation
            )
            clim = clim_range_per_gen[gen_key]
            cmap = "Magma"
        else:
            sliced = da.sel(year=year_or_stat, generation=generation)
            clim = clim_per_gen[gen_key]
            cmap = "Viridis"
        img = gv.Image(
            sliced, kdims=["longitude", "latitude"], vdims=[sliced.name]
        ).opts(
            cmap=cmap,
            alpha=alpha,
            colorbar=True,
            width=900,
            height=800,
            tools=["hover"],
            clim=clim,
        )
        tiles = gv.tile_sources.OSM.opts(alpha=1.0)
        # Add a red dot at the specified longitude/latitude
        point = gv.Points([(longitude, latitude)], kdims=["longitude", "latitude"]).opts(color="red", size=12, marker="o", line_color="black", nonselection_alpha=1.0)
        return tiles * img * point
    plot_pane = pn.bind(
        make_plot,
        year_or_stat=year_select,
        generation=gen_select,
        alpha=alpha_slider,
    )
    # Modern left menu with a card and a title
    menu = pn.Card(
        pn.pane.Markdown(
            f"<div style='color:#fff;font-size:18px;'><strong>{species} Dashboard</strong><br>"
            f"<b>Detection:</b> {detection_date}<br><b>Location:</b> ({latitude}, {longitude})</div>",
            sizing_mode='stretch_width'
        ),
        year_select,
        gen_select,
        alpha_slider,
        title="Options",
        width=180,
        margin=(20, 10, 20, 20),
        sizing_mode="stretch_height",
        css_classes=["fflies-dark-card"]
    )
    # Main layout: left menu, right plot
    layout = pn.Row(
        menu,
        pn.Column(plot_pane, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
        min_height=800
    )
    if save_path is not None:
        layout.save(save_path, embed=True)
    return layout
