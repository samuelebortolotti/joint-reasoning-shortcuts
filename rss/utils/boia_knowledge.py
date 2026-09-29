import torch

def get_boia_actions_from_concepts(concepts):
    # Column indices
    GREEN_LIGHT = 0
    FOLLOW = 1
    ROAD_CLEAR = 2
    RED_LIGHT = 3
    TRAFFIC_SIGN = 4
    CAR = 5
    PERSON = 6
    RIDER = 7
    OTHER_OBSTACLE = 8
    LEFT_LANE = 9
    LEFT_GREEN_LIGHT = 10
    LEFT_FOLLOW = 11
    NO_LEFT_LANE = 12
    LEFT_OBSTACLE = 13
    LEFT_SOLID_LINE = 14
    RIGHT_LANE = 15
    RIGHT_GREEN_LIGHT = 16
    RIGHT_FOLLOW = 17
    NO_RIGHT_LANE = 18
    RIGHT_OBSTACLE = 19
    RIGHT_SOLID_LINE = 20

    obstacle = concepts[:, CAR] | concepts[:, PERSON] | concepts[:, RIDER] | concepts[:, OTHER_OBSTACLE]
    road_clear = (~obstacle.bool()).to(torch.float32)
    move_forward = (concepts[:, GREEN_LIGHT].to(torch.bool) | concepts[:, FOLLOW].to(torch.bool) | road_clear.to(torch.bool)).to(torch.float32)
    stop = (concepts[:, RED_LIGHT] | concepts[:, TRAFFIC_SIGN] | obstacle).to(torch.float32)
    move_forward = move_forward * (~(stop.bool())).to(torch.float32)

    can_turn_left = (concepts[:, LEFT_LANE] | concepts[:, LEFT_GREEN_LIGHT] | concepts[:, LEFT_FOLLOW])
    cannot_turn_left = (concepts[:, NO_LEFT_LANE] | concepts[:, LEFT_OBSTACLE] | concepts[:, LEFT_SOLID_LINE])
    turn_left = (can_turn_left & (~cannot_turn_left)).to(torch.float32)

    can_turn_right = (concepts[:, RIGHT_LANE] | concepts[:, RIGHT_GREEN_LIGHT] | concepts[:, RIGHT_FOLLOW])
    cannot_turn_right = (concepts[:, NO_RIGHT_LANE] | concepts[:, RIGHT_OBSTACLE] | concepts[:, RIGHT_SOLID_LINE])
    turn_right = (can_turn_right & (~cannot_turn_right)).to(torch.float32)

    actions = torch.stack([move_forward, stop, turn_left, turn_right], dim=1)
    
    return actions
