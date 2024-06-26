REAL_COURT_KP = {
    "TOP_LEFT": [0, 0],
    "TOP_LEFT_HASH": [833, 0],
    "TOP_MID": [1400, 0],
    "TOP_RIGHT_HASH": [1967, 0],
    "TOP_RIGHT": [2800, 0],
    "RIGHT_FT_TOP_RIGHT": [2800, 505],
    "RIGHT_FT_TOP_LEFT": [2220, 505],
    "RIGHT_FT_BOTTOM_LEFT": [2220, 995],
    "RIGHT_FT_BOTTOM_RIGHT": [2800, 995],
    "BOTTOM_RIGHT": [2800, 1500],
    "BOTTOM_RIGHT_HASH": [1967, 1500],
    "BOTTOM_MID": [1400, 1500],
    "BOTTOM_LEFT_HASH": [833, 1500],
    "BOTTOM_LEFT": [0, 1500],
    "LEFT_FT_BOTTOM_LEFT": [0, 995],
    "LEFT_FT_BOTTOM_RIGHT": [580, 995],
    "LEFT_FT_TOP_RIGHT": [580, 505],
    "LEFT_FT_TOP_LEFT": [0, 505],
    "CENTER_TOP": [1400, 570],
    "CENTER_BOTTOM": [1400, 930],
    "VB_TOP_LEFT": [500, 300],
    "VB_TOP_LEFT_MID": [1100, 300],
    "VB_TOP_MID": [1400, 300],
    "VB_TOP_RIGHT_MID": [1700, 300],
    "VB_TOP_RIGHT": [2300, 300],
    "VB_BOTTOM_RIGHT": [2300, 1200],
    "VB_BOTTOM_RIGHT_MID": [1700, 1200],
    "VB_BOTTOM_MID": [1400, 1200],
    "VB_BOTTOM_LEFT": [500, 1200],
    "VB_BOTTOM_LEFT_MID": [1100, 1200],
}

VB_KP = [
    "VB_TOP_LEFT",
    "VB_TOP_LEFT_MID",
    "VB_TOP_MID",
    "VB_TOP_RIGHT_MID",
    "VB_TOP_RIGHT",
    "VB_BOTTOM_RIGHT",
    "VB_BOTTOM_RIGHT_MID",
    "VB_BOTTOM_MID",
    "VB_BOTTOM_LEFT",
    "VB_BOTTOM_LEFT_MID",
]

"""
Court lines and their points.
Used when searching for the largest visible polygon.
"""
LINES = {
    "CENTER_LINE": ["TOP_MID", "CENTER_TOP", "CENTER_BOTTOM", "BOTTOM_MID"],
    "LEFT_BASELINE": [
        "TOP_LEFT",
        "LEFT_FT_TOP_LEFT",
        "LEFT_FT_BOTTOM_LEFT",
        "BOTTOM_LEFT",
    ],
    "RIGHT_BASELINE": [
        "TOP_RIGHT",
        "RIGHT_FT_TOP_RIGHT",
        "RIGHT_FT_BOTTOM_RIGHT",
        "BOTTOM_RIGHT",
    ],
    "TOP_SIDELINE": [
        "TOP_LEFT",
        "TOP_LEFT_HASH",
        "TOP_MID",
        "TOP_RIGHT_HASH",
        "TOP_RIGHT",
    ],
    "BOTTOM_SIDELINE": [
        "BOTTOM_LEFT",
        "BOTTOM_LEFT_HASH",
        "BOTTOM_MID",
        "BOTTOM_RIGHT_HASH",
        "BOTTOM_RIGHT",
    ],
    "TOP_VB": [
        "VB_TOP_LEFT",
        "VB_TOP_LEFT_MID",
        "VB_TOP_MID",
        "VB_TOP_RIGHT_MID",
        "VB_TOP_RIGHT",
    ],
    "BOTTOM_VB": [
        "VB_BOTTOM_RIGHT",
        "VB_BOTTOM_RIGHT_MID",
        "VB_BOTTOM_MID",
        "VB_BOTTOM_LEFT",
        "VB_BOTTOM_LEFT_MID",
    ],
}
