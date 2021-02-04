package ticTacToe;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to.
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;                            //not 100 - see sirs video
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;                   //control policy - aa
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) throws IllegalMoveException
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);																		
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 * @throws IllegalMoveException 
	 */
	public QLearningAgent() throws IllegalMoveException
	{
		this(new RandomAgent(), 0.1, 10000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 * @throws IllegalMoveException 
	 */
	
	/* 
	 * REFERENCES - 
	 * 
	 * 1) Q-learning pseudocode -
	 * https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
	 * 
	 * 2) Epsilon Greedy -
	 * https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/#:~:text=Epsilon%2DGreedy%20Action%20Selection,a%20small%20chance%20of%20exploring.
	 * 
	 * 3) Get a key of a value in a map -
	 * https://www.baeldung.com/java-map-key-from-value
	 * 
	 * 4) Random double number - 
	 * https://www.tutorialspoint.com/generate-random-double-type-number-in-java#:~:text=In%20order%20to%20generate%20Random,from%20the%20random%20generator%20sequence.
	 * 
	 * */
	
	public void train() throws IllegalMoveException
	{
		
		for(int i=0; i<numEpisodes; i++) {
			
			Game gameStates = env.game;															//Get the current game state and store it in a variable
			
			Move movez = null;																	//Create an object of the move class and set it to null
			double qValue = 0.0;																//Variable to store the qValue		
			Double sample = 0.0;																//Variable to store the sample
			ArrayList<Double> gPrimeQValues = new ArrayList<Double>();							//Create an array list to store the q values of s'
			Double maxQValueGPrime = 0.0;														//Variable to store the max q value of the gPrimeQValues which contains the q values of s'
			
			
			
			while(!(gameStates.isTerminal())) {																//We loop until we reach a terminal state
				List<Move> m = gameStates.getPossibleMoves();												//Since we have to loop over the moves to get a random move or the best move of a particular game state, depending on epsilon, we get the possible moves for gameStates
				
				Random rando = new Random();
				double randDouble = rando.nextDouble();														//We generate a random double number
				//System.out.println(gameStates);												
				
				if(randDouble < epsilon) {																	//We compare if that random number is less than epsilon, if it is we explore
					//System.out.println("*********************************************");
					if(m.size()!=0) {
						Random rando1 = new Random();
						int num = rando1.nextInt(m.size());													
						movez = m.get(num);																		//We generate a random action and store it in movez
					}
					//System.out.println(movez);
					
				}
				else {																									//Else if that random number is greater than epsilon, we exploit
					ArrayList<Double> bestQValueList = new ArrayList<Double>();											//Create an array list to store the q values for that particular game state and move
					Double bestQValue = 0.0;																			//Create a variable to store the highest q value of a particular game state and move
					for(Move bestMove : m) {
						bestQValueList.add(qTable.getQValue(gameStates, bestMove));										//We add the q values to the list
						bestQValue = Collections.max(bestQValueList);													//We get the highest q value of the list and store it in the bestQValue variable
					}
					
					
				
					HashMap<Move,Double> moveMap = qTable.get(gameStates);												//We get the move and q value and store it in a map
			
					for(Map.Entry<Move, Double> bestMoveMap : moveMap.entrySet()) {										//We loop over the entry set of the moveMap
						if(bestMoveMap.getValue().equals(bestQValue)) {
							movez = bestMoveMap.getKey();																//We get the move with the highest q value
							//System.out.println("////////////");
						}
					}
					
					//System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
					bestQValueList.clear();																				//We clear the list for the next game state
						
				}
				
				m.clear();																								//We clear the move list since we got the move
				
				//System.out.println(movez);
				
				
					
				
						Outcome bestExperience = env.executeMove(movez);													//After executing the move we get the outcome for that move
						Game g = bestExperience.s;																			//We get the source state for that outcome		
						
						//System.out.println(bestMove);
						
						Game gPrime = bestExperience.sPrime;																//We get the target state for that outcome
						List<Move> mPrime = gPrime.getPossibleMoves();     													//We get the list of possible moves from the target state
						
						
						
						
						
								for(Move mo : mPrime) {
									
									
									
										gPrimeQValues.add(qTable.getQValue(gPrime, mo));										//We store the q values of the target state
										maxQValueGPrime = Collections.max(gPrimeQValues);										//We get the highest q value for that state 	
									
								}
								
								if(gPrime.isTerminal() == true) {
									maxQValueGPrime = 0.0;																			//If the target state is a terminal state, we set the highest q value of that state to 0.0
									
								}
								
						
						sample = bestExperience.localReward + (discount*maxQValueGPrime);											//Sample = R(s,a,s') + (gamma*max(Q(s',a')))
						
						//System.out.println(sample);
						//System.out.println(gameStates);
						
						Double oldEstimate = qTable.getQValue(g, movez);															//We get the q value of the source state
						
						//System.out.println(gameStates);
						
						 qValue = ( ((1-alpha)*oldEstimate) + (alpha*(sample)) );													//The formula = ( (1-alpha)*Q(s,a) + (alpha*(sample)) )
						 
						
						 
						 qTable.addQValue(g, movez, qValue);																				//We then add that q value into the qTable hashmap
						
					
					gPrimeQValues.clear();
				}
				env.resetEpisode();																											//We then reset the episode
				
		

		}
			
		

		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		Set<Game> gameSet = qTable.keySet();														//Since the game states are present as keys in the hashmap qTable, we get the states using the keySet and store them in a set
	
		HashMap<Game, Move> policyMap = new HashMap<Game,Move>();									//Creating a hashmap to store the policies
		
		for(Game gameStates : gameSet) {
			List<Move> m = gameStates.getPossibleMoves();											//We get the possible moves from that particular game state as we want to get the best move with the highest q value
			ArrayList<Double> bestQValueList = new ArrayList<Double>();								//Create an array list to store the q values for that particular game state and move
			Double bestQValue = 0.0;																//Create a variable to store the highest q value of a particular game state and move
			//if(m.size()!=0) {
				for(Move movez : m) {																//We loop over the list storing the possible moves
					bestQValueList.add(qTable.getQValue(gameStates, movez));						//We add the q values to the list
					bestQValue = Collections.max(bestQValueList);									//We get the highest q value of the list and store it in the bestQValue variable								
				}
			//}	
			Move bestMove = null;																	//We create a game object to store the best move (move with the highest q value)
			
			HashMap<Move,Double> moveMap = qTable.get(gameStates);									//We get the move and q value and store it in a map
			
			for(Map.Entry<Move, Double> bestMoveMap : moveMap.entrySet()) {							//We loop over the entry set of the moveMap
				if(bestMoveMap.getValue().equals(bestQValue)) {
					bestMove = bestMoveMap.getKey();												//We get the move with the highest q value
				}
			}
			policyMap.put(gameStates, bestMove);													//We put the game state and the best move inside the policyMap hashmap
			bestQValueList.clear();																	//We then clear the list
			
		}
		
		Policy policy = new Policy(policyMap);														//Create an object of the Policy class and provide the hashmap (that stores the policies) created above as an argument for the Policy class constructor
		return policy;																				//return the policy
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
