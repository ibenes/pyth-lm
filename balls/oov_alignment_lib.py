import numpy as np


def word_distance(a, b):
    return 0 if a == b else 1


DIAGONAL_MOVE = 0
HORIZONAL_MOVE = 1
VERTICAL_MOVE = -1


class AlignmentExtractor:
    def __init__(self):
        pass

    def extract(self, a, b, moves_taken):
        self._a = a
        self._b = b
        self._ptr_a = moves_taken.shape[0] - 1
        self._ptr_b = moves_taken.shape[1] - 1
        self._alignment = []
        self._words_a = []
        self._words_b = []
        while self._ptr_a != 0 or self._ptr_b != 0:
            move = moves_taken[self._ptr_a, self._ptr_b]

            if move == VERTICAL_MOVE:
                self._vertical_move()
            elif move == HORIZONAL_MOVE:
                self._horizontal_move()
            else:
                self._diagonal_move()
                if self._ptr_a == 0 or self._ptr_b == 0:
                    continue  # no flushing
                self._flush()

        self._flush()

        return list(reversed(self._alignment))

    def _flush(self):
        self._alignment.append((list(reversed(self._words_a)), list(reversed(self._words_b))))
        self._words_a = []
        self._words_b = []

    def _vertical_move(self):
        self._ptr_a -= 1
        self._words_a.append(self._a[self._ptr_a])

    def _horizontal_move(self):
        self._ptr_b -= 1
        self._words_b.append(self._b[self._ptr_b])

    def _diagonal_move(self):
        if self._ptr_a >= 1:
            self._words_a.append(self._a[self._ptr_a-1])
        if self._ptr_b >= 1:
            self._words_b.append(self._b[self._ptr_b-1])

        self._ptr_a -= 1
        self._ptr_b -= 1


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

    # assert(alignment[0] == ([], []))
    # alignment = alignment[1:]

    # print(partial_costs)
    # print(moves_taken)
    return AlignmentExtractor().extract(a, b, moves_taken)
