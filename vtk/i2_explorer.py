import pandas as pd
import numpy as np

import param
import holoviews as hv
import panel as pn
from holoviews.operation.datashader import rasterize
import hvplot.pandas

from utilities import *
from data_loading import run_options, parcel_labels, default_parcels, parcel_labels_map, I2Run

pn.extension('vtk', defer_load=True, loading_indicator=False)

cmap = 'seismic'

class I2Explorer(pn.viewable.Viewer):
    
    # Parametrized instance of I2Run for data loading
    run = param.ClassSelector(class_=I2Run, doc="Instance of I2Run to load data for selected run")

    # Parameters for user selections
    run_select = param.Selector(objects=run_options, label="Select a run", doc="Prefix of the I2 run")
    parcels_select = param.ListSelector(default=default_parcels, objects=parcel_labels, label="Select parcels", doc="Brain parcellations according to Schaefer 100 atlas")
    parcels_dgfc_vol = param.Parameter(doc="Filtered data for selected parcels")
    parcels_dfc_matrix = param.Parameter(doc="Dynamic functional connectivity matrix for selected parcels")

    frame_toggle = param.ObjectSelector(default='Range', objects=['Range', 'Player'])
    frame_range = param.Range((1,100), bounds=(1,100), label="Selected frames")
    frame_player = param.Integer(1, bounds=(1, 100), label="Selected frame")

    i_slice = param.Integer(97//2, bounds=(1, 97), label="i slice")
    j_slice = param.Integer(115//2, bounds=(1, 115), label="j slice")
    k_slice = param.Integer(97//2, bounds=(1, 97), label="k slice")
    center_slice = param.Action(lambda self: self.reset_slices(), label="Center slices")

    
    def __init__(self, **params):
        super().__init__(**params)
        pn.state.onload(self.update_run)

        self.run_select_widget = pn.widgets.Select.from_param(self.param.run_select)
        self.parcels_select_widget = pn.widgets.MultiSelect.from_param(self.param.parcels_select, size=8)
        
        self.frame_toggle_widget = pn.widgets.RadioButtonGroup.from_param(self.param.frame_toggle)
        self.frame_range_widget = pn.widgets.IntRangeSlider.from_param(self.param.frame_range, visible=True)
        self.frame_player_widget = pn.widgets.Player.from_param(self.param.frame_player, value=0, loop_policy='loop', show_value=True, width=310, visible=False)

        
        self.i_slice_widget = pn.widgets.IntSlider.from_param(self.param.i_slice)
        self.j_slice_widget = pn.widgets.IntSlider.from_param(self.param.j_slice)
        self.k_slice_widget = pn.widgets.IntSlider.from_param(self.param.k_slice)
        self.center_slice_widget = pn.widgets.Button.from_param(self.param.center_slice)
        

    @param.depends('run_select', watch=True)
    def update_run(self):
        self.run = I2Run(self.run_select)
        
        self.fc_clim = normalize_to_symmetric_range(self.run.dfc_matrix.min(), self.run.dfc_matrix.min())
        self.gfc_clim = normalize_to_symmetric_range(self.run.dgfc_matrix.min(), self.run.dgfc_matrix.max())
        
        self.update_parcels()

        self.param.frame_player.bounds = (1, self.run.n_windows)
        self.param.frame_range.bounds = (1, self.run.n_windows)
        self.param.frame_range = (1, self.run.n_windows)
        self.param.i_slice.bounds = (1, self.run.run_img.shape[0])
        self.param.j_slice.bounds = (1, self.run.run_img.shape[1])
        self.param.k_slice.bounds = (1, self.run.run_img.shape[2])

        self.frame_player = 1
        self.update_frame()
        

    @param.depends('parcels_select', watch=True)
    def update_parcels(self):
        
        parcels_idx = np.array([parcel_labels_map[p] for p in self.parcels_select])

        # Update gdfc volume
        self.parcels_vol_mask = np.isin(self.run.parcel_img_resampled.get_fdata(), parcels_idx)
        parcels_d_vol_mask = np.broadcast_to(self.parcels_vol_mask[..., None], self.run.dgfc_vol.shape)
        self.parcels_dgfc_vol = np.where(parcels_d_vol_mask, self.run.dgfc_vol, np.nan)

        # Update dfc & gdfc matrix
        self.parcels_dfc_matrix = self.run.dfc_matrix[:,parcels_idx-1][:,:,parcels_idx-1]
        self.parcels_dgfc_matrix = self.run.dgfc_matrix[:,parcels_idx-1]


    @param.depends('run_select', 'parcels_select', 'frame_toggle', 'frame_range', 'frame_player', watch=True)
    def update_frame(self):
        if self.frame_toggle == 'Player':
            self.frame_vol = self.parcels_dgfc_vol[..., self.frame_player - 1]
            self.frame_matrix = self.parcels_dfc_matrix[self.frame_player - 1]
        
        elif self.frame_toggle == 'Range':
            self.frame_vol = self.parcels_dgfc_vol[..., self.frame_range[0]-1:self.frame_range[1]-1].mean(axis=3)
            self.frame_matrix = self.parcels_dfc_matrix[self.frame_range[0]-1:self.frame_range[1]-1].mean(axis=0)
    
    @param.depends('run_select', 'parcels_select', 'frame_toggle', 'frame_range', 'frame_player', 'i_slice', 'j_slice', 'k_slice')
    def slices_display(self):
        slice_common = dict(
            vol = self.frame_vol,
            low = self.gfc_clim[0],
            high = self.gfc_clim[1],
            cmap = cmap,
            colorbar = True,
            tools = ['hover']
        )
        
        dmap_common = dict(
            width = 400,
            height = 350
        )

        # dmap_i = rasterize(hv.DynamicMap(pn.bind(image_slice_i, si=self.i_slice, **slice_common)).opts(title='slice i', **dmap_common))
        dmap_i = hv.DynamicMap(pn.bind(image_slice_i_with_underlay, self.i_slice, slice_common)).opts(title='slice i', **dmap_common)
        dmap_j = hv.DynamicMap(pn.bind(image_slice_j_with_underlay, self.j_slice, slice_common)).opts(title='slice j', **dmap_common)
        dmap_k = hv.DynamicMap(pn.bind(image_slice_k_with_underlay, self.k_slice, slice_common)).opts(title='slice k', **dmap_common)

        # fix for common colorbar here likely here... or i'll have to build it to also match the heatmap
        return (dmap_i + dmap_j + dmap_k).opts(shared_axes=True, title='Slice Views of Parcellated Global FC')
        

    @param.depends('run_select','parcels_select', 'i_slice', 'j_slice', 'k_slice')
    def volume_display(self):
        # arr = np.where(self.selected_parcels_mask_3d, 1., np.nan) # nan option
        # arr = self.parcels_vol_mask.astype(float) # bool option
    
        volume = pn.pane.VTKVolume(
                underlay_vol,  orientation_widget=True,
                display_slices=True, display_volume=False,
                colormap='Grayscale', render_background='#ffffff', 
                # diffuse=0.5

        )
        volume.slice_i = self.i_slice
        volume.slice_j = self.j_slice
        volume.slice_k = self.k_slice
        
        return volume
    
    
    @param.depends('run_select', 'parcels_select', 'frame_toggle', 'frame_range', 'frame_player')
    def dfc_matrix_display(self):
        dfc_df = pd.DataFrame(self.frame_matrix, index=self.parcels_select, columns=self.parcels_select)
        dfc_heatmap = dfc_df.hvplot.heatmap(
            title=f'fc matrix', flip_yaxis=True, clim=self.fc_clim
        ).opts(
            cmap=cmap, xrotation=45, yrotation=45, 
            height=450, width=550
        )

        return dfc_heatmap


    @param.depends('run_select', 'parcels_select', 'frame_toggle', 'frame_range', 'frame_player')
    def lineplot_display(self):
        # Set up data for GFC lineplot
        y_parcels = pd.DataFrame(self.parcels_dgfc_matrix, columns=self.parcels_select)
        y_parcels['time'] = self.run.window_timestamps
        y_parcels = y_parcels.melt(id_vars='time', var_name='parcel', value_name='gdfc')

        # Set up data for functional ratings scatterplot
        functional_questions = ['Positive/Negative','Acceptance/Resistance','No Insight/Strong Insight']
        functional_ratings = self.run.ratings_df[self.run.ratings_df.Question.isin(functional_questions)]

        # Create hvplot elements
        lineplot = y_parcels.hvplot.line(x='time', y='gdfc', by='parcel', alpha=0.1, legend=False) * \
                    y_parcels.groupby('time').gdfc.mean().reset_index().hvplot.line(x='time', y='gdfc', color='black', label='Mean GFC')

        scatterplot = functional_ratings.hvplot.scatter(x='Seconds since start', y='Answer', by='Question', legend='top_left')
        
        if self.frame_toggle == 'Player':
            current_time = self.run.window_timestamps[self.frame_player - 1]
            time_overlay = hv.DynamicMap(pn.bind(vertical_line_callback, x=current_time))
        elif self.frame_toggle == 'Range':
            start_time = self.run.window_timestamps[self.frame_range[0] - 1]
            end_time = self.run.window_timestamps[self.frame_range[1] - 1]
            time_overlay = hv.DynamicMap(pn.bind(transparent_box_callback, x0=start_time, x1=end_time))

        final_lineplot = (lineplot * scatterplot * time_overlay).opts(
            title="GFC of selected parcels and subjective ratings with Current Time",
            xlabel="Time (s)",
            ylabel="GFC",
            ylim=(-0.3, 0.5),
            hooks=[overlay_hook],
            legend_position='bottom',
        ).opts(height=400, width=650)
        return final_lineplot


    def reset_slices(self):
        """Set slices to center of selected parcels."""

        # This is a hacky way to find the center... will improve later
        indices = np.where(self.parcels_vol_mask)
        centroid = np.mean(indices, axis=0)

        self.i_slice = int(centroid[0])
        self.j_slice = int(centroid[1])
        self.k_slice = int(centroid[2])
        
    @param.depends('run_select', 'parcels_select')
    def header(self):
        """Dynamically updates the markdown block with current selections."""
        return pn.pane.Markdown(f"""
            ### Run Metadata
            - **Run prefix**: {self.run.run_prefix} 
            - **Ratings prefix**: {self.run.ratings_prefix}
            - **Window size**: {self.run.window_size}
            - **Volume shape**: {self.parcels_dgfc_vol.shape}
            - **Number of selected parcels**: {len(self.parcels_select)}
            - **Sum values of selected parcels**: {self.parcels_dgfc_vol.sum()}
            - **Selected parcels**: {', '.join(self.parcels_select)}
        """, sizing_mode="stretch_width")

    
    @param.depends('frame_toggle', watch=True)
    def update_frame_widget(self):
        """Show only one of the widgets (Player or IntRangeSlider) based on the toggle selection."""
        if self.frame_toggle == 'Player':
            self.frame_player_widget.visible = True
            self.frame_range_widget.visible = False
        elif self.frame_toggle == 'Range':
            self.frame_player_widget.visible = False
            self.frame_range_widget.visible = True
    
    
    def __panel__(self):

        template = pn.template.ReactTemplate(
            title='I2 Visualization Toolkit',
            favicon='favicon.ico',
            prevent_collision=False,
            row_height=125,
            
        )
        template.sidebar.append(
            pn.Column(
                self.run_select_widget,
                self.parcels_select_widget,
                pn.Spacer(height=20),
                
                pn.panel('## Time controller'),
                self.frame_toggle_widget,
                self.frame_range_widget,
                self.frame_player_widget,
                pn.Spacer(height=20),
                
                pn.panel('## Slice controller'),
                pn.Column(self.i_slice_widget, self.j_slice_widget, self.k_slice_widget),
                self.center_slice_widget,
            )
        )

        template.main[:2,:7] = self.header
        template.main[:2,7:10] = self.volume_display
        template.main[2:5,:11] = self.slices_display
        template.main[5:9,:5] = self.dfc_matrix_display
        template.main[5:9,5:11] = self.lineplot_display
        
        return template


I2Explorer().servable()