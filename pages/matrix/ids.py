# matrix/ids.py
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixIds:
    # ABC
    a_score: str = "a_score_id"
    b_score: str = "b_score_id"
    c_score: str = "c_score_id"

    # XYZ
    x_score: str = "x_score_id"
    y_score: str = "y_score_id"
    z_score: str = "z_score_id"

    # ROP/SS
    lead_time: str = "lead_time_id_for_matix"
    service_ratio: str = "servis_ratio_id_for_matrix"

    # Filters
    gr_ms: str = "gr_multyselect_id_for_matrix"
    cat_ms: str = "cat_multyselect_id_for_matrix"
    groupby_sc: str = "groupby_sc_id_for_matrix"

    # Buttons
    launch: str = "launch_batton_id_for_matrix"

    # Help modals + open buttons
    abc_help_open: str = "abc_help_open_id_for_matrix"
    abc_help_modal: str = "abc_help_modal_id_for_matrix"

    xyz_help_open: str = "xyz_help_open_id_for_matrix"
    xyz_help_modal: str = "xyz_help_modal_id_for_matrix"

    rop_help_open: str = "rop_help_open_id_for_matrix"
    rop_help_modal: str = "rop_help_modal_id_for_matrix"

    filter_help_open: str = "filter_help_open_id_for_matrix"
    filter_help_modal: str = "filter_help_modal_id_for_matrix"
    
    zones_help_open: str = "zones_help_open_id_for_matrix"
    zones_help_modal: str = "zones_help_modal_id_for_matrix"


@dataclass(frozen=True)
class MatrixRightIds:
    right_container: str = "right_conteiner_id_for_matrix"
    matrix_grid: str = "matrix-ag-greed-id"

    barcode_drawer: str = "barcode_drawer_id"
    barcode_drawer_body: str = "barcode_drawer_body_id"
    loading: str = "matrix_loading"

    # download (RIGHT side)
    download_btn: str = "matrix_download_excel_btn"
    download: str = "matrix_download_excel"
    
    #  CSV download (FAST)
    download_csv_btn: str = "matrix_download_csv_btn"
    download_csv: str = "matrix_download_csv"
    
    # manufacturer filter (RIGHT header)
    manu_ms: str = "matrix_manu_ms"
    manu_badge: str = "matrix_manu_badge"
    store: str = "matrix_store"


    @property
    def content(self) -> str:
        return f"{self.right_container}-content"

    

