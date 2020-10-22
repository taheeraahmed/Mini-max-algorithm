# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util


from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a game_state and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentgame_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        game_states (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a game_state (pacman.py)
        successorgame_state = currentgame_state.generatePacmanSuccessor(action)
        newPos = successorgame_state.getPacmanPosition()
        newFood = successorgame_state.getFood()
        newGhostStates = successorgame_state.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorgame_state.getScore()

def scoreEvaluationFunction(currentgame_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentgame_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.getLegalActions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generateSuccessor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.getNumAgents():
        Returns the total number of agents in the game

        game_state.isWin():
        Returns whether or not the game state is a winning state

        game_state.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.miniMax(game_state, 0, 0)[1]
        
    def miniMax(self, game_state, agent_index, depth):
        #Check if depth is reached or if the game is over
        if(self.depth == depth or game_state.isWin() or game_state.isLose()):
            #returns static evaluation
            return self.evaluationFunction(game_state), None
        
        #If the agent is pacman (agent_index = 0)
        if agent_index == 0:
            #Finding pacmans legal actions
            legal_actions_pacman = game_state.getLegalActions(0)
            #Crating a list to store all the possible scores in
            pacman_scores = []
            #Looping through pacmans legal actions 
            for action in legal_actions_pacman:
                #Generate children
                pacman_child = game_state.generateSuccessor(agent_index, action)
                #Calls miniMax recursivly with all pacmans children
                temp_score = self.miniMax(pacman_child, 1, depth)[0]
                #Adds a childs score to scores
                pacman_scores.append(temp_score)
            
            #Find the highest score in pacman_scores
            highest_pacman_score = max(pacman_scores)
            #Choose pacmans action based on the highest_pacman_score 
            action = legal_actions_pacman[pacman_scores.index(highest_pacman_score)]
            return highest_pacman_score, action
        
        # The agent is a ghost
        else:
            #Finding the ghost's legal actions
            legal_actions_ghost = game_state.getLegalActions(agent_index)
            #Crating a list to store all possible the scores in
            ghost_scores = []
            lowest_ghost_score = 0
            #Creating a list to store the indices belonging to the lowest_score in ghosts_scores
            lowest_ghost_score_indices = []
            #Number of ghosts
            num_ghosts = game_state.getNumAgents() - 1
            #Check for last ghost
            if agent_index == num_ghosts:
                for action in legal_actions_ghost:
                    #Generate children
                    ghost_child = game_state.generateSuccessor(agent_index, action)
                    #Calls miniMax recursivly with all ghost's children - but also changing depth and switching to pacman
                    temp_ghost_score = self.miniMax(ghost_child, 0, depth + 1)[0]
                    #Adds a childs score to scores
                    ghost_scores.append(temp_ghost_score)
            else:
                #Looping through the ghost's legal actions 
                for action in legal_actions_ghost:
                    #Generate children
                    ghost_child = game_state.generateSuccessor(agent_index, action)
                    #Calls miniMax recursivly on the next ghost with all ghost's children 
                    temp_ghost_score = self.miniMax(ghost_child, agent_index + 1, depth)[0]
                    #Adds a childs score to scores
                    ghost_scores.append(temp_ghost_score)
            
            #Find the lowest score in scores
            lowest_ghost_score = min(ghost_scores)
            #Choose ghosts action based on the lowest_ghost_score 
            action = legal_actions_ghost[ghost_scores.index(lowest_ghost_score)]
            return lowest_ghost_score, action

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    Dersom det er en maks node og alpha er større enn beta; prune aka break
    """
    
    def alphaBeta(self, game_state, depth, agent_index, alpha, beta):
        
        num_ghosts = game_state.getNumAgents() - 1
        
        if num_ghosts < agent_index:
            depth += 1
            agent_index = 0
            
        #Check if depth is reached or if the game is over
        if(self.depth == depth or game_state.isWin() or game_state.isLose()):
            #returns static evaluation
            return self.evaluationFunction(game_state), None   
        
        
        
        #If Pacman, use maxValue
        if agent_index == 0: 
            return self.maxValue(game_state, depth, agent_index, alpha, beta)

        #Not pacman means ghost, use minValue
        else: 
            return self.minValue(game_state, depth, agent_index, alpha, beta)

    def maxValue(self, game_state, depth, agent_index, alpha, beta):
        max_value = float('-inf')
        scores = []
        legal_actions = game_state.getLegalActions(agent_index)
        for action in legal_actions:
            #Generate children
            child = game_state.generateSuccessor(agent_index, action)
            #Calls minValue on maxnodes child because its a min value node
            temp_score = self.alphaBeta(child, depth, agent_index + 1, alpha, beta)[0]
            scores.append(temp_score)
            #If a childs value is larger than the max_value -- change the max_value
            max_value = max(max_value, temp_score)
            
            #Pruning a max node by comparing it to beta (MINs best option
            if max_value > beta:
                return max_value, action
            #replace alpha value if temp_score is larger than the current alpha value --> better path for MAX
            alpha = max(alpha, max_value)
        highest_score = max(scores)
        action = legal_actions[scores.index(highest_score)]
        return highest_score, action 

    def minValue(self, game_state, depth, agent_index, alpha, beta):
        min_value = float('inf')
        scores = []
        legal_actions = game_state.getLegalActions(agent_index)
        for action in legal_actions:
            #Generate children
            child = game_state.generateSuccessor(agent_index, action)
            #Calls max recursivly with all children, because we are on min node, and we know all children will be min-nodes
            temp_score = self.alphaBeta(child, depth, agent_index +1 , alpha, beta)[0]
            scores.append(temp_score)
            #If a childs value is smaller than the min_value -- change the min_value
            min_value = min(min_value, temp_score)
            
            #Pruning a max node by comparing it to alpha (MAXs best option)
            if min_value < alpha:
                return min_value, action
                
            #replace beta value if temp_score is less than the current beta-value --> better path for MIN
            beta = min(beta, min_value)

        lowest_score = min(scores)
        action = legal_actions[scores.index(lowest_score)]
        return lowest_score, action
            
    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphaBeta(game_state, 0, 0, float('-inf'), float('inf'))[1]
        #util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentgame_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
