from enum import Enum
from pathlib import Path
BASE_DIR = Path(__file__).parent
BASE_DIR = BASE_DIR / "templates/icons"
from typing import Literal

SIZES = {
    'xs':11,
    'sm':13,
    'md':16,
    'lg':18,
    'xl':21,
    '1em':'1em'
}

class Streamline(Enum):
    shipment_star = "streamline/streamline-ultimate-color--shipment-star.svg"
    office_file_xls = "streamline/streamline-ultimate-color--office-file-xls.svg"
    business_deal_handshake_1 = "streamline/streamline-ultimate-color--business-deal-handshake-1.svg"
    graphic_tablet_draw = "streamline/streamline-ultimate-color--graphic-tablet-draw.svg"
    smiley_sad_1 = "streamline/streamline-ultimate-color--smiley-sad-1.svg"
    easter_egg = "streamline/streamline-stickies-color--easter-egg.svg"
    check = "streamline/streamline-ultimate-color--check.svg"
    attachment = "streamline/streamline-ultimate-color--attachment.svg"
    analytics_pie_2 = "streamline/streamline-ultimate-color--analytics-pie-2.svg"
    compass_1 = "streamline/streamline-ultimate-color--compass-1.svg"
    concert_rock_1 = "streamline/streamline-ultimate-color--concert-rock-1.svg"
    single_neutral_circle = "streamline/streamline-ultimate-color--single-neutral-circle.svg"
    data_file_bars_edit = "streamline/streamline-ultimate-color--data-file-bars-edit.svg"
    science_physics_law = "streamline/streamline-ultimate-color--science-physics-law.svg"
    notes_paper_text = "streamline/streamline-ultimate-color--notes-paper-text.svg"
    graph_stats_descend = "streamline/streamline-ultimate-color--graph-stats-descend.svg"
    champagne_cooler = "streamline/streamline-ultimate-color--champagne-cooler.svg"
    time_duo = "streamline/streamline-stickies-color--time-duo.svg"
    analytics_bars_3d = "streamline/streamline-ultimate-color--analytics-bars-3d.svg"
    safety = "streamline/streamline-stickies-color--safety.svg"
    calendar_date = "streamline/streamline-ultimate-color--calendar-date.svg"
    office_file_graph = "streamline/streamline-ultimate-color--office-file-graph.svg"
    love = "streamline/streamline-stickies-color--love.svg"
    android_setting_duo = "streamline/streamline-stickies-color--android-setting-duo.svg"
    module_puzzle = "streamline/streamline-ultimate-color--module-puzzle.svg"
    optimization_graph = "streamline/streamline-ultimate-color--optimization-graph.svg"
    business_rabbit_hat_1 = "streamline/streamline-ultimate-color--business-rabbit-hat-1.svg"
    compass_directions = "streamline/streamline-ultimate-color--compass-directions.svg"
    receipt_dollar = "streamline/streamline-ultimate-color--receipt-dollar.svg"
    star = "streamline/streamline-stickies-color--star.svg"
    love_duo = "streamline/streamline-stickies-color--love-duo.svg"
    help_duo = "streamline/streamline-stickies-color--help-duo.svg"
    drugs_cannabis = "streamline/streamline-ultimate-color--drugs-cannabis.svg"
    style_one_pin_check = "streamline/streamline-ultimate-color--style-one-pin-check.svg"
    bus_route_info = "streamline/streamline-stickies-color--bus-route-info.svg"
    remove_bold = "streamline/streamline-ultimate-color--remove-bold.svg"
    graph_pie = "streamline/streamline-stickies-color--graph-pie.svg"
    eiffel_tower_duo = "streamline/streamline-stickies-color--eiffel-tower-duo.svg"
    key = "streamline/streamline-stickies-color--key.svg"
    backpack_duo = "streamline/streamline-stickies-color--backpack-duo.svg"
    reciept_1_duo = "streamline/streamline-stickies-color--reciept-1-duo.svg"
    task_list_to_do = "streamline/streamline-ultimate-color--task-list-to-do.svg"
    wand = "streamline/streamline-stickies-color--wand.svg"
    check_badge = "streamline/streamline-ultimate-color--check-badge.svg"
    book_star = "streamline/streamline-ultimate-color--book-star.svg"
    validation_1_duo = "streamline/streamline-stickies-color--validation-1-duo.svg"
    labtop_duo = "streamline/streamline-stickies-color--labtop-duo.svg"
    wireless_payment_credit_card_dollar = "streamline/streamline-ultimate-color--wireless-payment-credit-card-dollar.svg"
    rating_star = "streamline/streamline-ultimate-color--rating-star.svg"
    human_resources_team_settings = "streamline/streamline-ultimate-color--human-resources-team-settings.svg"
    reciept_1 = "streamline/streamline-stickies-color--reciept-1.svg"
    help = "streamline/streamline-stickies-color--help.svg"
    smiley_lol_sideways = "streamline/streamline-ultimate-color--smiley-lol-sideways.svg"
    data_file_search = "streamline/streamline-ultimate-color--data-file-search.svg"
    ghost = "streamline/streamline-stickies-color--ghost.svg"
    analytics_board_graph_line = "streamline/streamline-ultimate-color--analytics-board-graph-line.svg"
    arrow_double_up = "streamline/streamline-ultimate-color--arrow-double-up.svg"
    on_off_1_duo = "streamline/streamline-stickies-color--on-off-1-duo.svg"
    bug_duo = "streamline/streamline-stickies-color--bug-duo.svg"
    messages_bubble_square_typing_1 = "streamline/streamline-ultimate-color--messages-bubble-square-typing-1.svg"
    performance_money_decrease = "streamline/streamline-ultimate-color--performance-money-decrease.svg"
    book_book_pages = "streamline/streamline-ultimate-color--book-book-pages.svg"
    cash_payment_bill = "streamline/streamline-ultimate-color--cash-payment-bill.svg"
    common_file_edit = "streamline/streamline-ultimate-color--common-file-edit.svg"
    delivery_truck_cargo = "streamline/streamline-ultimate-color--delivery-truck-cargo.svg"
    performance_increase = "streamline/streamline-ultimate-color--performance-increase.svg"
    rating_star_winner = "streamline/streamline-ultimate-color--rating-star-winner.svg"
    muslim_duo = "streamline/streamline-stickies-color--muslim-duo.svg"
    smiley_smile_1 = "streamline/streamline-ultimate-color--smiley-smile-1.svg"
    backpack = "streamline/streamline-stickies-color--backpack.svg"
    gauge_dashboard = "streamline/streamline-ultimate-color--gauge-dashboard.svg"
    bus_route_info_duo = "streamline/streamline-stickies-color--bus-route-info-duo.svg"
    send_email_envelope = "streamline/streamline-ultimate-color--send-email-envelope.svg"
    library_research_duo = "streamline/streamline-stickies-color--library-research-duo.svg"
    common_file_text = "streamline/streamline-ultimate-color--common-file-text.svg"
    bug = "streamline/streamline-stickies-color--bug.svg"
    wand_duo = "streamline/streamline-stickies-color--wand-duo.svg"
    labtop = "streamline/streamline-stickies-color--labtop.svg"
    control = "streamline/streamline-stickies-color--control.svg"
    safety_duo = "streamline/streamline-stickies-color--safety-duo.svg"
    sun = "streamline/streamline-stickies-color--sun.svg"
    on_off_1 = "streamline/streamline-stickies-color--on-off-1.svg"
    balloon_tour_duo = "streamline/streamline-stickies-color--balloon-tour-duo.svg"
    dangerous_chemical_lab = "streamline/streamline-stickies-color--dangerous-chemical-lab.svg"
    design_tool_magic_wand = "streamline/streamline-ultimate-color--design-tool-magic-wand.svg"
    star_duo = "streamline/streamline-stickies-color--star-duo.svg"
    multiple_users_1 = "streamline/streamline-ultimate-color--multiple-users-1.svg"
    notes_book = "streamline/streamline-ultimate-color--notes-book.svg"
    picture_sun = "streamline/streamline-ultimate-color--picture-sun.svg"
    ranking_stars_ribbon = "streamline/streamline-ultimate-color--ranking-stars-ribbon.svg"
    task_list_approve = "streamline/streamline-ultimate-color--task-list-approve.svg"
    wrench_duo = "streamline/streamline-stickies-color--wrench-duo.svg"
    arrow_double_down_1 = "streamline/streamline-ultimate-color--arrow-double-down-1.svg"
    business_deal_cash_2 = "streamline/streamline-ultimate-color--business-deal-cash-2.svg"
    pile_of_money_duo = "streamline/streamline-stickies-color--pile-of-money-duo.svg"
    balloon_tour = "streamline/streamline-stickies-color--balloon-tour.svg"
    graph_pie_duo = "streamline/streamline-stickies-color--graph-pie-duo.svg"
    cash_payment_bills_1 = "streamline/streamline-ultimate-color--cash-payment-bills-1.svg"
    road_sign_stop = "streamline/streamline-ultimate-color--road-sign-stop.svg"
    send_email_fly = "streamline/streamline-ultimate-color--send-email-fly.svg"
    business_big_small_fish = "streamline/streamline-ultimate-color--business-big-small-fish.svg"
    cog = "streamline/streamline-ultimate-color--cog.svg"
    party_confetti = "streamline/streamline-ultimate-color--party-confetti.svg"
    face_id_1_duo = "streamline/streamline-stickies-color--face-id-1-duo.svg"
    shopping_cart_full = "streamline/streamline-ultimate-color--shopping-cart-full.svg"
    validation_1 = "streamline/streamline-stickies-color--validation-1.svg"
    
    def render(self, size: Literal['xs','sm','md','lg','xl','1em'] = '1em',  inline=True):
        svg_path = BASE_DIR / self.value
        svg_content = ''
        icon_size = SIZES[size] if isinstance(size,str) else size
        if inline:
            svg_content = svg_path.read_text(encoding="utf-8")
            svg_content = " ".join(svg_content.split())
            for s in [16, 20, 24, 32, 40, 48, 64]:
                svg_content = svg_content.replace(f'width="{s}"', f'width="{icon_size}"')
                svg_content = svg_content.replace(f'height="{s}"', f'height="{icon_size}"')
            return svg_content
        else:
            # возвращает тег <img>
            return f'<img src="file://{svg_path}" width="{icon_size}" height="{icon_size}" style="vertical-align: middle;">'
            

class ColorEmoji(Enum):
    pile_of_poo = "coloremoji/streamline-emojis--pile-of-poo.svg"
    anxious_face = "coloremoji/streamline-emojis--anxious-face.svg"
    face_with_monocle = "coloremoji/streamline-emojis--face-with-monocle.svg"
    wrench = "coloremoji/streamline-emojis--wrench.svg"
    face_savoring_food = "coloremoji/streamline-emojis--face-savoring-food.svg"
    face_with_raised_eyebrow = "coloremoji/streamline-emojis--face-with-raised-eyebrow.svg"
    beating_heart = "coloremoji/streamline-emojis--beating-heart.svg"
    balloon = "coloremoji/streamline-emojis--balloon.svg"
    face_blowing_a_kiss = "coloremoji/streamline-emojis--face-blowing-a-kiss.svg"
    bird_2 = "coloremoji/streamline-emojis--bird-2.svg"
    confused_face = "coloremoji/streamline-emojis--confused-face.svg"
    backhand_index_pointing_right_2 = "coloremoji/streamline-emojis--backhand-index-pointing-right-2.svg"
    backhand_index_pointing_down_1 = "coloremoji/streamline-emojis--backhand-index-pointing-down-1.svg"
    airplane = "coloremoji/streamline-emojis--airplane.svg"
    beaming_face_with_smiling_eyes = "coloremoji/streamline-emojis--beaming-face-with-smiling-eyes.svg"
    backhand_index_pointing_right_1 = "coloremoji/streamline-emojis--backhand-index-pointing-right-1.svg"
    nauseated_face_2 = "coloremoji/streamline-emojis--nauseated-face-2.svg"
    writing_hand_1 = "coloremoji/streamline-emojis--writing-hand-1.svg"
    heart_with_arrow = "coloremoji/streamline-emojis--heart-with-arrow.svg"
    backhand_index_pointing_down_2 = "coloremoji/streamline-emojis--backhand-index-pointing-down-2.svg"
    cow_face = "coloremoji/streamline-emojis--cow-face.svg"
    birthday_cake_2 = "coloremoji/streamline-emojis--birthday-cake-2.svg"
    face_with_rolling_eyes = "coloremoji/streamline-emojis--face-with-rolling-eyes.svg"
    birthday_cake_3 = "coloremoji/streamline-emojis--birthday-cake-3.svg"
    calendar = "coloremoji/streamline-emojis--calendar.svg"
    bar_chart = "coloremoji/streamline-emojis--bar-chart.svg"
    maple_leaf = "coloremoji/streamline-emojis--maple-leaf.svg"
    watermelon_1 = "coloremoji/streamline-emojis--watermelon-1.svg"
    zzz = "coloremoji/streamline-emojis--zzz.svg"
    exclamation_mark = "coloremoji/streamline-emojis--exclamation-mark.svg"
    package = "coloremoji/streamline-emojis--package.svg"
    pill = "coloremoji/streamline-emojis--pill.svg"
    determined_face = "coloremoji/streamline-emojis--determined-face.svg"
    dollar_banknote = "coloremoji/streamline-emojis--dollar-banknote.svg"
    high_voltage = "coloremoji/streamline-emojis--high-voltage.svg"
    artist_palette = "coloremoji/streamline-emojis--artist-palette.svg"
    flashlight = "coloremoji/streamline-emojis--flashlight.svg"
    backhand_index_pointing_up_1 = "coloremoji/streamline-emojis--backhand-index-pointing-up-1.svg"
    crazy_face = "coloremoji/streamline-emojis--crazy-face.svg"
    battery = "coloremoji/streamline-emojis--battery.svg"
    blossom = "coloremoji/streamline-emojis--blossom.svg"
    clinking_glasses_4 = "coloremoji/streamline-emojis--clinking-glasses-4.svg"
    double_exclamation_mark = "coloremoji/streamline-emojis--double-exclamation-mark.svg"
    boar_1 = "coloremoji/streamline-emojis--boar-1.svg"
    crossed_fingers_1 = "coloremoji/streamline-emojis--crossed-fingers-1.svg"
    beer_mug = "coloremoji/streamline-emojis--beer-mug.svg"
    elephant = "coloremoji/streamline-emojis--elephant.svg"
    goat = "coloremoji/streamline-emojis--goat.svg"
    baby_chick = "coloremoji/streamline-emojis--baby-chick.svg"
    cactus_1 = "coloremoji/streamline-emojis--cactus-1.svg" 
    
    
    def render(self, size: Literal['xs','sm','md','lg','xl','1em'] = '1em',  inline=True):
        svg_path = BASE_DIR / self.value
        svg_content = ''
        icon_size = SIZES[size] if isinstance(size,str) else size
        if inline:
            svg_content = svg_path.read_text(encoding="utf-8")
            svg_content = " ".join(svg_content.split())
            for s in [16, 20, 24, 32, 40, 48, 64]:
                svg_content = svg_content.replace(f'width="{s}"', f'width="{icon_size}"')
                svg_content = svg_content.replace(f'height="{s}"', f'height="{icon_size}"')
            return svg_content
        else:
            # возвращает тег <img>
            return f'<img src="file://{svg_path}" width="{icon_size}" height="{icon_size}" style="vertical-align: middle;">'   
        
class Solar(Enum):
    box_outline = "solar/solar--box-outline.svg"
    star_fall_2_bold = "solar/solar--star-fall-2-bold.svg"
    paperclip_bold = "solar/solar--paperclip-bold.svg"
    course_up_outline = "solar/solar--course-up-outline.svg"
    archive_minimalistic_bold = "solar/solar--archive-minimalistic-bold.svg"
    chat_round_like_outline = "solar/solar--chat-round-like-outline.svg"
    arrow_right_outline = "solar/solar--arrow-right-outline.svg"
    cup_star_bold = "solar/solar--cup-star-bold.svg"
    star_fall_minimalistic_2_bold = "solar/solar--star-fall-minimalistic-2-bold.svg"
    check_read_outline = "solar/solar--check-read-outline.svg"
    course_down_outline = "solar/solar--course-down-outline.svg"
    settings_bold = "solar/solar--settings-bold.svg"
    arrow_up_bold = "solar/solar--arrow-up-bold.svg"
    hashtag_bold = "solar/solar--hashtag-bold.svg"
    course_up_bold = "solar/solar--course-up-bold.svg"
    arrow_down_bold = "solar/solar--arrow-down-bold.svg"
    arrow_right_bold = "solar/solar--arrow-right-bold.svg"
    course_down_bold = "solar/solar--course-down-bold.svg"
    check_circle_outline = "solar/solar--check-circle-outline.svg"
    check_square_outline = "solar/solar--check-square-outline.svg"
    cat_outline = "solar/solar--cat-outline.svg"
    star_angle_bold = "solar/solar--star-angle-bold.svg"
    magic_stick_3_bold = "solar/solar--magic-stick-3-bold.svg"
    sledgehammer_bold = "solar/solar--sledgehammer-bold.svg"
    diagram_up_bold = "solar/solar--diagram-up-bold.svg"
    chat_round_like_bold = "solar/solar--chat-round-like-bold.svg"
    airbuds_case_charge_outline = "solar/solar--airbuds-case-charge-outline.svg"
    bar_chair_outline = "solar/solar--bar-chair-outline.svg"
    diagram_down_bold = "solar/solar--diagram-down-bold.svg"
    graph_down_bold = "solar/solar--graph-down-bold.svg"
    map_arrow_down_outline = "solar/solar--map-arrow-down-outline.svg"
    star_rainbow_bold = "solar/solar--star-rainbow-bold.svg"
    document_text_bold = "solar/solar--document-text-bold.svg"
    arrow_down_outline = "solar/solar--arrow-down-outline.svg"
    paperclip_outline = "solar/solar--paperclip-outline.svg"
    diagram_down_outline = "solar/solar--diagram-down-outline.svg"
    arrow_up_outline = "solar/solar--arrow-up-outline.svg"
    fire_bold = "solar/solar--fire-bold.svg"
    cat_bold = "solar/solar--cat-bold.svg"
    balloon_bold = "solar/solar--balloon-bold.svg"
    chart_outline = "solar/solar--chart-outline.svg"
    flame_bold = "solar/solar--flame-bold.svg"
    map_arrow_up_outline = "solar/solar--map-arrow-up-outline.svg"
    diagram_up_outline = "solar/solar--diagram-up-outline.svg"
    bug_outline = "solar/solar--bug-outline.svg"

    
    def render(self, 
               size: Literal['xs','sm','md','lg','xl','1em'] = '1em',
               c: Literal ["red", "green", "blue", "black", "white", "gray","orange", "yellow", "purple", "pink", "brown", "cyan", "magenta"] | None = None,   
               inline=True):
        svg_path = BASE_DIR / self.value
        color = c
        svg_content = ''
        icon_size = SIZES[size] if isinstance(size,str) else size
        if inline:
            svg_content = svg_path.read_text(encoding="utf-8")
            svg_content = " ".join(svg_content.split())
            for s in [16, 20, 24, 32, 40, 48, 64, 50]:
                svg_content = svg_content.replace(f'width="{s}"', f'width="{icon_size}"')
                svg_content = svg_content.replace(f'height="{s}"', f'height="{icon_size}"')
            if color:
               svg_content = svg_content.replace('fill="currentColor"',f'fill="{color}"') 
            return svg_content
        else:
            # возвращает тег <img>
            return f'<img src="file://{svg_path}" width="{icon_size}" height="{icon_size}" style="vertical-align: middle;">'  

  




