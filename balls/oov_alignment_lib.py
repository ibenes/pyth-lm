import numpy as np


def word_distance(a, b):
    return 0 if a == b else 1


DIAGONAL_MOVE = 0
HORIZONAL_MOVE = 1
VERTICAL_MOVE = -1


def align(a, b):
    local_costs = np.zeros(shape=(len(a)+1, len(b)+1))

    for i, w_a in enumerate(a):
        for j, w_b in enumerate(b):
            local_costs[i+1, j+1] = word_distance(w_a, w_b)

    partial_costs = np.full(shape=(len(a)+1, len(b)+1), fill_value=np.inf)
    moves_taken = np.full(shape=(len(a)+1, len(b)+1), fill_value=np.inf)

    for i in range(partial_costs.shape[0]):
        for j in range(partial_costs.shape[1]):
            if i == 0 and j == 0:
                partial_costs[i, j] = 0.0
                moves_taken[i, j] = DIAGONAL_MOVE
            elif i == 0 and j != 0:
                partial_costs[i, j] = partial_costs[i, j-1] + 1.0
                moves_taken[i, j] = HORIZONAL_MOVE
            elif i != 0 and j == 0:
                partial_costs[i, j] = partial_costs[i-1, j] + 1.0
                moves_taken[i, j] = VERTICAL_MOVE
            else:
                vertical_cost = partial_costs[i-1, j] + 1.0
                horizontal_cost = partial_costs[i, j-1] + 1.0
                diagonal_cost = partial_costs[i-1, j-1] + local_costs[i, j]

                best_cost = min([vertical_cost, horizontal_cost, diagonal_cost])

                partial_costs[i, j] = best_cost
                if best_cost == vertical_cost:
                    moves_taken[i, j] = VERTICAL_MOVE
                elif best_cost == horizontal_cost:
                    moves_taken[i, j] = HORIZONAL_MOVE
                else:
                    moves_taken[i, j] = DIAGONAL_MOVE

    ptr_a = moves_taken.shape[0] - 1
    ptr_b = moves_taken.shape[1] - 1
    alignment = []
    words_a = []
    words_b = []
    while ptr_a != 0 or ptr_b != 0:
        move = moves_taken[ptr_a, ptr_b]

        if move == VERTICAL_MOVE:
            ptr_a -= 1
            words_a.append(a[ptr_a])
        elif move == HORIZONAL_MOVE:
            ptr_b -= 1
            words_b.append(b[ptr_b])
        else:
            if ptr_a >= 1:
                words_a.append(a[ptr_a-1])
            if ptr_b >= 1:
                words_b.append(b[ptr_b-1])

            ptr_a -= 1
            ptr_b -= 1

            alignment.append((list(reversed(words_a)), list(reversed(words_b))))
            words_a = []
            words_b = []

    alignment = list(reversed(alignment))
    # assert(alignment[0] == ([], []))
    # alignment = alignment[1:]

    print(partial_costs)
    print(moves_taken)
    return alignment
