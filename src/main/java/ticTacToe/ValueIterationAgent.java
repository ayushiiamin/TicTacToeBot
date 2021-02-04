package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();				
	
	
	/**
	 * the discount factor
	 */
	double discount=0.9;       
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();					
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=50;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);								
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	/* 
	 * REFERENCES - 
	 * 
	 * 1) Various methods of Map -
	 * https://docs.oracle.com/javase/8/docs/api/java/util/Map.html
	 * 
	 * 2) Get the max value in an ArrayList -
	 * https://www.javaprogramto.com/2020/04/java-arraylist-maximum-value.html
	 * 
	 * 
	 * */
	
	
	
	public void iterate()
	{
		for(int i=0; i<k; i++) {
			
			Set<Game> vg = valueFunction.keySet();							//Since the game states are present as keys in the hashmap valueFunction, we get the states using the keySet and store them in a set
				for(Game gameStates : vg) {             					//We loop over each game state, as for each game we want to update V(g)
					
					if(gameStates.isTerminal() == true) {					//Checking if the game is a terminal state, if it is put the v(g) value as 0.0 for that game state
						valueFunction.put(gameStates, 0.0);
						continue;
					}
					
					List<Move> m = gameStates.getPossibleMoves();                            		//Since we want to maximize over all moves from that particular Game state g, we have to go over all the possible moves from g
					ArrayList<Double> viListGS = new ArrayList<Double>();							//Array List to store the VI values (vkg) of that particular game state
					Double maxVIListGS = 0.0; 														//Variable to store the max VI value of the viListGS which contains the VI values of that particular game state
					
					for(Move mo : m) {													
						List<TransitionProb> interimState = mdp.generateTransitions(gameStates, mo);				//Get the transitions of the Q state				
						double vkg = 0.0;														                   //Variable To store the sum (transitionProbability * (reward + (gamma*V(s'))))
						
						for(TransitionProb transitionOutcomes : interimState) {							//This loop does the summing of (transitionProbability * (reward + (gamma*V(s')))) and it loops over all Outcomes from the Q-state  
							
							//Outcome - T(S,A,R,S') 
							
							double transitionFunc = transitionOutcomes.prob;							//The transition probability - T(S,A,S')
							double reward = transitionOutcomes.outcome.localReward;						//The reward - R(S,A,S')
							Game gPrime = transitionOutcomes.outcome.sPrime;						    //the Vk(s')
							double vkgPrime = valueFunction.get(gPrime);								//the value of Vk(s') HashMap<Game, Value>
							
							vkg += (transitionFunc * (reward + (discount * vkgPrime)));					//The summation of (transitionProbability * (reward + (gamma*V(s')))) for all moves in that particular game state
						}
						viListGS.add(vkg);																//we add the vkg values into the VI list
						maxVIListGS = Collections.max(viListGS);										//we get the max value present in that list and store it in the variable that stores the max VI value
					}
					viListGS.clear();      																				//We clear the VI List after iterating over that particular game state
					valueFunction.put(gameStates, maxVIListGS);																	//We then update Vk(g)
				}
		}
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		Set<Game> vg = valueFunction.keySet();															//Since the game states are present as keys in the hashmap valueFunction, we get the states using the keySet and store them in a set
		
		HashMap<Game, Move> policyMap = new HashMap<Game,Move>();										//Creating a hashmap to store the policy
		
		for(Game gameStates : vg) {																		//We loop over each game state, as for each game we want to update V(g)
			List<Move> m = gameStates.getPossibleMoves();												//Since we want to maximize over all moves from that particular Game state g, we have to go over all the possible moves from g
			ArrayList<Double> viListGS = new ArrayList<Double>();										//Array List to store the VI values (vkg) of that particular game state
			Double maxVIListGS = 0.0;																	//Variable to store the max VI value of the viListGS which contains the VI values of that particular game state
			
			for(Move mo : m) {
				List<TransitionProb> interimState = mdp.generateTransitions(gameStates, mo);				//Get the transitions of the Q state
				double vkg = 0.0;																			//Variable To store the sum (transitionProbability * (reward + (gamma*V(s'))))
				
				for(TransitionProb transitionOutcomes : interimState) {										//This loop does the summing of (transitionProbability * (reward + (gamma*V(s')))) and it loops over all Outcomes from the Q-state
					double transitionFunc = transitionOutcomes.prob;										//The transition probability - T(S,A,S')
					double reward = transitionOutcomes.outcome.localReward;									//The reward - R(S,A,S')
					Game gPrime = transitionOutcomes.outcome.sPrime;										//the Vk(s')
					double vkgPrime = valueFunction.get(gPrime);											//the value of Vk(s') HashMap<Game, Value>
					
					vkg += (transitionFunc * (reward + (discount * vkgPrime)));								//The summation of (transitionProbability * (reward + (gamma*V(s')))) for all moves in that particular game state
				}
				viListGS.add(vkg);																			//we add the vkg values into the VI list
				maxVIListGS = Collections.max(viListGS);													//we get the max value present in that list and store it in the variable that stores the max VI value
				
				if(maxVIListGS == vkg) {																	//performing the one-step expectimax by checking if the max value is equal to the vkg value that has been calculated
					policyMap.put(gameStates, mo);															//If it is, then put that specific game state and move into the hashmap created to store the policies
				}
			}
			viListGS.clear();																				//We clear the VI List after iterating over that particular game state
		}
		Policy policy = new Policy(policyMap);																//Create an object of the Policy class and provide the hashmap (that stores the policies) created above as an argument for the Policy class constructor
		return policy;																						//return the policy
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
