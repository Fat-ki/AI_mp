import heapq


def best_first_search(starting_state):
    # TODO(III): You should copy your code from MP3 here
    visited_states = {starting_state: (None, 0)}

    frontier = []
    heapq.heappush(frontier, starting_state)

    while frontier:
        cur_state = heapq.heappop(frontier)
        if cur_state.is_goal():
            return backtrack(visited_states, cur_state)
        neighbor_state = cur_state.get_neighbors()
        for n_state in neighbor_state:
            n_distance = n_state.dist_from_start
            if n_state not in visited_states or n_distance < visited_states[n_state][1]:
                visited_states[n_state] = (cur_state, n_distance)
                heapq.heappush(frontier, n_state)
    return []


def backtrack(visited_states, goal_state):
    # TODO(III): You should copy your code from MP3 here
    path = []

    cur_state = goal_state
    while cur_state is not None:
        path.append(cur_state)
        cur_state = visited_states[cur_state][0]
    path.reverse()

    return path
