"""Same evaluator/protocol/output seal as V10, changing only the backend."""
import run_v10
from joint_decoder import filter_frozen
if __name__=='__main__':
    run_v10.filter_frozen=filter_frozen
    run_v10.main()
