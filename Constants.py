
class Constants:
    COOLDOWN_SWAP = 3
    COOLDOWN_GENERATE = 5
    COOLDOWN_SCORE = 1
    COOLDOWN_TELE_GATE = 5
    COOLDOWN_TELE_QUBIT = 5

    ENTANGLEMENT_PROBABILITY = 0.95 

    REWARD_STOP = -20 
    REWARD_FOR_SWAP = -5
    REWARD_SCORE = 500  
    REWARD_INVALID_ACTION = -100

    NUMQ = 18  # number of qubits for the random dag/circuit
    NUMG = 50  # number of gates for the random dag/circuit

    REWARD_EMPTY_DAG = 3000 
    REWARD_DEADLINE = -3000

    # files for inference or training if training from file is true
    DAG_FILE = "datasets/circuits/18Qubits_30Gates.json"
    MAPPING_FILE = "datasets/mappings/18QubitsOn32QubitMappingV2.json"


    DISTANCE_MULT = 18  # the multiplier for the difference of total distances metric
    DISTANCE_QUANTUM_LINK = 30 # virtual distance for quantum links (cross processor)
    DISTANCE_BETWEEN_EPR = 1
