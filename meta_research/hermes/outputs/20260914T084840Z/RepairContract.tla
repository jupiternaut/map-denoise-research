---- MODULE RepairContract ----
\* Contracts only. Finite 3-point tape.
\* - records append-only (query writes unused slots; retest may overwrite a slot)
\* - query/retest billed
\* - isolate free and reversible, never deletes rec
\* - PolicyPick uses rec only (Hidden is a constant, not a policy input)

EXTENDS Integers, FiniteSets

CONSTANTS X, MaxC, MaxClock
None == -1

VARIABLES rec, isol, cost, pick, clock

vars == <<rec, isol, cost, pick, clock>>

TypeOK ==
  /\ rec \in [X -> -1..2]
  /\ isol \subseteq X
  /\ cost \in 0..MaxC
  /\ pick \in X \cup {None}
  /\ clock \in 0..MaxClock

Queried == {x \in X : rec[x] # None}

Hidden == [x \in X |-> 1]
Dirt   == [x \in X |-> 0]

Init ==
  /\ rec = [x \in X |-> None]
  /\ isol = {}
  /\ cost = 0
  /\ pick = None
  /\ clock = 0

PolicyPick ==
  IF \E x \in X : rec[x] = None
  THEN CHOOSE x \in X : rec[x] = None
  ELSE CHOOSE x \in X : TRUE

Query ==
  /\ cost < MaxC
  /\ \E x \in X :
       /\ rec[x] = None
       /\ x = PolicyPick
       /\ rec' = [rec EXCEPT ![x] = Hidden[x]]
       /\ cost' = cost + 1
       /\ pick' = x
       /\ UNCHANGED <<isol, clock>>

Retest ==
  /\ cost < MaxC
  /\ Queried # {}
  /\ \E x \in Queried :
       /\ rec' = [rec EXCEPT ![x] = Dirt[x]]
       /\ cost' = cost + 1
       /\ pick' = x
       /\ UNCHANGED <<isol, clock>>

Isolate ==
  /\ clock < MaxClock
  /\ \E x \in Queried :
       /\ x \notin isol
       /\ isol' = isol \cup {x}
       /\ pick' = x
       /\ clock' = clock + 1
       /\ UNCHANGED <<rec, cost>>

Unisolate ==
  /\ clock < MaxClock
  /\ \E x \in isol :
       /\ isol' = isol \ {x}
       /\ pick' = x
       /\ clock' = clock + 1
       /\ UNCHANGED <<rec, cost>>

Done ==
  /\ \/ cost = MaxC
     \/ clock = MaxClock
  /\ UNCHANGED vars

Next == Query \/ Retest \/ Isolate \/ Unisolate \/ Done

Spec == Init /\ [][Next]_vars

IsolateSubsetQueried == isol \subseteq Queried

Safety ==
  /\ TypeOK
  /\ IsolateSubsetQueried
====
