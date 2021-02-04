package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	
	/* 
	 * REFERENCES - 
	 * 
	 * 1) Random element from a list
	 * https://www.geeksforgeeks.org/randomly-select-items-from-a-list-in-java/
	 * https://www.baeldung.com/java-random-list-element
	 * 
	 * */
	public void initRandomPolicy()
	{
		List<Game> g = Game.generateAllValidGames('X');								//Initially, we generate all valid games where X is playing and store it in a list
		//System.out.println(curPolicy);
		for(Game ga : g) {															//we loop over the list
			//System.out.println(ga);
			List<Move> m = ga.getPossibleMoves();										//we get the possible moves from that game state
			//System.out.println(m);
			if(m.size()!=0) {															//We check whether the size is not equal to 0, as there are some empty elements ([]) present in the list and they are invalid moves
				Random rando = new Random();
				//System.out.println(rando);
				int num = rando.nextInt(m.size());										//we generate a random integer which will act as the index of the element present in the list
		        Move move = m.get(num);													//We get a random move, using the random integer who is the index of the element to be returned
		        //System.out.println(move);
		        	
		        		curPolicy.put(ga, move);										//we store that particular game and the random move in curPolicy
		        	
		        
		        //System.out.println(curPolicy);
		       // System.out.println(m);
			}
			//System.out.println(curPolicy);
			//System.out.println(m);
			m.clear();																	//Clear the possible moves from game ga, so that we can store the possible moves from the next game state (next iteration)
		}
		
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	
	
	/* 
	 * REFERENCES - 
	 * 
	 * 1) Convert negative number to positive
	 * https://mkyong.com/java/how-to-convert-negative-number-to-positive-in-java/
	 * https://stackoverflow.com/questions/493494/make-a-negative-number-positive
	 * 
	 * */
	
	protected void evaluatePolicy(double delta)
	{
		
		
		double maxValue = 99999.0;													//We assign the max value to a huge number in order to run the do while loop			
		double prevValue = 0.0;						
		double updateValue = 0.0;													//we assign the previous value and update value to 0.0
		
		//System.out.println("The maxValue is " + maxValue);
		
		ArrayList<Double> updateValueList = new ArrayList<Double>();					//Creating a list to store the updates
		
		do {                               
			
			
			
			Set<Game> vg = curPolicy.keySet();													//We get the game states (using keSet() as Game acts as the key in the map) from the current policy map and store it in a set
			
			//prevValue = policyValues.get(vg);
			
			
				for(Game gameStates : vg) {														//We loop over the game states
					
					if(gameStates.isTerminal() == true) {											//Checking if the game is a terminal state, if it is put the vpi(g) value as 0.0 for that game state
						policyValues.put(gameStates, 0.0);
						continue;
					}
					
					
					
					
					prevValue = policyValues.get(gameStates);																//We get the value that is mapped to the current game state and that is considered as the previous value
					Move m = curPolicy.get(gameStates);																		//We get the move to which the current game state is mapped to. We do this as in policy evaluation, the action is fixed
					
					//System.out.println("The value of prevValue " + policyValues.get(gameStates));
					
					//System.out.println(m);
						
				
							List<TransitionProb> interimState = mdp.generateTransitions(gameStates, m);							//Get the transitions of the Q stat
							double vkg = 0.0;																					//Variable To store the sum (transitionProbability * (reward + (gamma*Vpi(s'))))
							
							
							
							for(TransitionProb transitionOutcomes : interimState) {													//This loop does the summing of (transitionProbability * (reward + (gamma*Vpi(s')))) and it loops over all Outcomes from the Q-state
								
								
								
								double transitionFunc = transitionOutcomes.prob;												//The transition probability - T(S,pi(S),S')
								double reward = transitionOutcomes.outcome.localReward;											//The reward - R(S,pi(S),S')
								Game gPrime = transitionOutcomes.outcome.sPrime;												//the Vk(s')	
								double vkgPrime = policyValues.get(gPrime);														//the value of Vk(s') HashMap<Game, Value>
								
								//System.out.println("The value of s' " + vkgPrime);
								
								vkg += (transitionFunc * (reward + (discount * vkgPrime)));										//The summation of (transitionProbability * (reward + (gamma*Vpi(s')))) 
								
								//System.out.println(vkg);
							}
							
							//System.out.println("The value of prevValue " + prevValue);
							//System.out.println("The value of vkg " + vkg);
							
							updateValue = Math.abs(prevValue - vkg);															//We calculate the update and take the absolute value of it as we want positive values here 
							//updateValue = (prevValue - vkg);
							
							//System.out.println("The value of updateValue " + updateValue);
							
							updateValueList.add(updateValue);																		//After calculating we store the updates in the list
							maxValue = Collections.max(updateValueList);															//We get the max update in the list. We do this in order to check for convergence
							
							policyValues.put(gameStates, vkg);																		//then insert the current game state and the Vpi(g)	value	
							
							//System.out.println("The value of vkg " + vkg);
							//System.out.println("The value of maxValue " + maxValue);
					
				}
				updateValueList.clear();																								//clear the update list, in order to store the update values in it, at the next iteration 
				
		}while(maxValue >= delta);   																									//run this loop until convergence. If (maxValue < delta) then convergence has taken place
		
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		
		
		Set<Game> vg = curPolicy.keySet();																	//We get the game states (using keSet() as Game acts as the key in the map) from the current policy map and store it in a set
		int flag=0;																							//Initializing flag =0
		
		
		
		for(Game gameStates : vg) {																			//We loop over the game states
		
			List<Move> gameMoves = gameStates.getPossibleMoves();											//we get the possible moves from that game state
			ArrayList<Double> piListGS = new ArrayList<Double>();											//Array List to store the policy extraction values (vkg) of that particular game state	
			Double maxPIListGS = 0.0;																		//Variable to store the max policy extraction value of the piListGS which contains the policy extraction values of that particular game state
			Move newMove = null;																			//Creating a variable to store the move that improves the current policy
			
			
			
				for(Move mo : gameMoves) {
					List<TransitionProb> interimState = mdp.generateTransitions(gameStates, mo);			//Get the transitions of the Q state
					double vkg = 0.0;																		//Variable To store the sum (transitionProbability * (reward + (gamma*V(s'))))
					
					
					
					for(TransitionProb transitionOutcomes : interimState) {											//This loop does the summing of (transitionProbability * (reward + (gamma*V(s')))) and it loops over all Outcomes from the Q-state
						double transitionFunc = transitionOutcomes.prob;											//The transition probability - T(S,A,S')
						double reward = transitionOutcomes.outcome.localReward;										//The reward - R(S,A,S')	
						Game gPrime = transitionOutcomes.outcome.sPrime;											//the V(s')
						double vkgPrime = policyValues.get(gPrime);													//the value of V(s') HashMap<Game, Value>
						
						
						
						vkg += (transitionFunc * (reward + (discount * vkgPrime)));										//The summation of (transitionProbability * (reward + (gamma*V(s')))) for all moves in that particular game state
					}
					piListGS.add(vkg);																					//we add the vkg values into the policy extraction list
					maxPIListGS = Collections.max(piListGS);															//we get the max value present in that list and store it in the variable that stores the max policy extraction value
					
					
					
					if(maxPIListGS == vkg) {          																	//performing the one-step expectimax by checking if the max value is equal to the vkg value that has been calculated
						newMove = mo;																					//If it is, then assign the current move as the move that improves the current policy
					}
					
					
				}
				piListGS.clear();																						//We clear the policy extraction List after iterating over that particular game state
				
				double curValue = policyValues.get(gameStates);															//Creating a variable to store the current value of that particular game state					
				
				if(maxPIListGS>curValue) {																				//Checking if the max policy extraction value is greater than the current value	
					curPolicy.put(gameStates, newMove);																	//If it is, then put the current game state and the move that improves the current policy, into the hashmap that stores the current policy
					flag=1;																								//set the flag to be equal to 1
				}
			
				
		}
		if(flag==1) {
			return true;																								//This returns true only when the policy has changed and improved
		}
		else {
			return false;																								//If this method returns false, that means the policy wasn't improved and convergence has taken place
		}
		
		
		
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		
		while(improvePolicy()==true) {																				//We run the loop till the policy improves
			this.evaluatePolicy(delta);
		}
		super.policy=new Policy(curPolicy);	
		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}

