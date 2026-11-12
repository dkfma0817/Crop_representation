import sys
import src.pacoattr.boot as B
run=sys.argv[1];B.RUN=run+'/mean';sys.argv=['boot','--split','test','--conds','P_linear,M_linear,B0_linear,C0_linear','--B','1000','--evaldir','eval_test','--procs','24','--out',run+'/mean/boot_test.json'];B.main()
