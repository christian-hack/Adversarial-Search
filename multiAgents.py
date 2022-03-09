# multiAgents.py
# --------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # unless ghosts are near, Pac-Man should focus solely on eating food
        food_Dist = float("inf")
        newFood = successorGameState.getFood().asList()
        for food in newFood:
            food_Dist = min(food_Dist, manhattanDistance(newPos, food))
        ghostPositions = successorGameState.getGhostPositions()
        # loop through list of ghost positions to determine manhattan distance from Pac-Man current position to ghost
        for ghost in ghostPositions:
            if(manhattanDistance(newPos, ghost) < 2):
                return -float("inf")
        return successorGameState.getScore() + 1.0/food_Dist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.maxValue(gameState, 0, 0)[0]

    #Minimax
    def miniMax(self, gameState, agentIndex, depth):
        #Checks to see if the depth has been reached, also if the current state is winning/losing
        if depth is self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[1]
        else:
            return self.minValue(gameState, agentIndex, depth)[1]

    #Maximum
    def maxValue(self, gameState, agentIndex, depth):
        #Sets basis for optimal (maximum) action
        optAction = ("max", -float("inf"))
        #Runs through all actions
        for action in gameState.getLegalActions(agentIndex):
            #Finds the successor action
            succAction = (action, self.miniMax(gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), 
            depth + 1))
            #Finds maximum
            optAction = max(optAction, succAction, key = lambda x:x[1])
        return optAction

    #Minimum
    def minValue(self, gameState, agentIndex, depth):
        #Sets basis for optimal (minimum) action
        optAction = ("min", float("inf"))
        #Loops through all actions
        for action in gameState.getLegalActions(agentIndex):
            #Finds the successor action
            succAction = (action, self.miniMax(gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), 
            depth + 1))
            #Finds minimum
            optAction = min(optAction, succAction, key = lambda x:x[1])
        return optAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.maxValue(-float("inf"), float("inf"), 0, 0, gameState)[0]

    def alphaBeta(self, alpha, beta, depth, index, gameState):
        # ensure depth isn't reached and that current state is not winning/losing state
        if depth is self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index is 0:
            return self.maxValue(alpha, beta, depth, index, gameState)[1]
        else:
            return self.minValue(alpha, beta, depth, index, gameState)[1]

    def minValue(self, alpha, beta, depth, index, gameState):
        # set basis for optimal (minimum) action
        optAction = ("min", float("inf"))
        # loop through all actions
        for action in gameState.getLegalActions(index):
            succAction = (action, self.alphaBeta(alpha, beta, depth+1, (depth+1) % gameState.getNumAgents(), gameState.generateSuccessor(index, action)))
            # find MIN
            optAction = min(optAction, succAction, key = lambda x:x[1])
            if optAction[1] < alpha:
                return optAction
            else:
                beta = min(beta, optAction[1])
        return optAction
    def maxValue(self, alpha, beta, depth, index, gameState):
        # set basis for optimal (maximum) action
        optAction = ("max", -float("inf"))
        # loop through all actions
        for action in gameState.getLegalActions(index):
            succAction = (action, self.alphaBeta(alpha, beta, depth+1, (depth+1) % gameState.getNumAgents(), gameState.generateSuccessor(index, action)))
            # find MAX
            optAction = max(optAction, succAction, key = lambda x:x[1])
            if optAction[1] > beta:
                return optAction
            else:
                alpha = max(alpha, optAction[1])
        return optAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectiMax(gameState, "expect", maxDepth, 0)[0]
 
    #Expectimax
    def expectiMax(self, gameState, action, depth, agentIndex):
        #Checks if the depth has been reached, or if the current state is winning/losing
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (action, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxValue(gameState, action, depth, agentIndex)
        else: 
            return self.expValue(gameState, action, depth, agentIndex)

    #Maximum        
    def maxValue(self, gameState, action, depth, agentIndex):
        #Sets basis for to optimal (maximum) valie
        optAction = ('max', -(float('inf')))
        #Loops through valid actions
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = action
            #Checks depth
            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
                #Finds successor value 
                succValue = self.expectiMax(gameState.generateSuccessor(agentIndex, legalAction), succAction, depth -1, nextAgent)
                #Finds the maximum action
                optAction = max(optAction, succValue, key = lambda x:x[1])
                return optAction

    #Exp value
    def expValue(self, gameState, action, depth, agentIndex):
        #Obtains valid actions
        legalActions = gameState.getLegalActions(agentIndex)
        #Sets probability to 1/the length of actions
        probability = 1.0/len(legalActions)
        #Sets the average score to 0
        avScore = 0
        #Loops through actions
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            #Finds the optimal action (expectiMax)
            optAction = self.expectiMax(gameState.generateSuccessor(agentIndex, legalAction), action, depth -1, nextAgent)
            avScore += optAction[1] * probability
        return (action, avScore)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
