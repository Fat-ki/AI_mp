# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from state import MazeState
from maze import Maze
def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    start_state = maze.get_start()
    visited_states = {start_state: (None, 0)}

    frontier = []
    heapq.heappush(frontier, start_state)

    while frontier:
        cur_state = heapq.heappop(frontier)
        if cur_state.is_goal():
            return backtrack(visited_states, cur_state)
        #print(f"current: {cur_state.state},")
        #print(f"queue:{frontier}")
        neighbor_state = cur_state.get_neighbors()
        for n_state in neighbor_state:
            n_distance = n_state.dist_from_start
            if n_state not in visited_states or n_distance < visited_states[n_state][1]:
                visited_states[n_state] = (cur_state, n_distance)
                heapq.heappush(frontier, n_state)
    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []

    cur_state = current_state
    while cur_state is not None:
        path.append(cur_state)
        cur_state = visited_states[cur_state][0]
    path.reverse()

    return path
