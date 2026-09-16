---- MODULE RepairContract ----
\* Finite 3-point tape. Checked state invariants only (see TLA_CLAIMS.md).
\* orig is first-write-only. cur may change on retest. isolate does not
\* delete orig. Query/retest billed. Isolate/unisolate use a clock bound
\* so TLC terminates; that bound is an artifact, not a research claim.

EXTENDS Integers, FiniteSets

CONSTANTS X, MaxC, MaxClock
None == -1

VARIABLES orig, cur, isol, cost, pick, clock

vars == <<orig, cur, isol, cost, pick, clock>>

TypeOK ==
  /\ orig \in [X -> -1..2]
  /\ cur  \in [X -> -1..2]
  /\ isol \subseteq X
  /\ cost \in 0..MaxC
  /\ pick \in X \cup {None}
  /\ clock \in 0..MaxClock

QueriedOrig == {x \in X : orig[x] # None}

Hidden == [x \in X |-> 1]
Dirt   == [x \in X |-> 0]

Init ==
  /\ orig = [x \in X |-> None]
  /\ cur  = [x \in X |-> None]
  /\ isol = {}
  /\ cost = 0
  /\ pick = None
  /\ clock = 0

PolicyPick ==
  IF \E x \in X : orig[x] = None
  THEN CHOOSE x \in X : orig[x] = None
  ELSE CHOOSE x \in X : TRUE

Query ==
  /\ cost < MaxC
  /\ \E x \in X :
       /\ orig[x] = None
       /\ x = PolicyPick
       /\ orig' = [orig EXCEPT ![x] = Hidden[x]]
       /\ cur'  = [cur  EXCEPT ![x] = Hidden[x]]
       /\ cost' = cost + 1
       /\ pick' = x
       /\ UNCHANGED <<isol, clock>>

Retest ==
  /\ cost < MaxC
  /\ QueriedOrig # {}
  /\ \E x \in QueriedOrig :
       /\ orig' = orig
       /\ cur'  = [cur EXCEPT ![x] = Dirt[x]]
       /\ cost' = cost + 1
       /\ pick' = x
       /\ UNCHANGED <<isol, clock>>

Isolate ==
  /\ clock < MaxClock
  /\ \E x \in QueriedOrig :
       /\ x \notin isol
       /\ isol' = isol \cup {x}
       /\ pick' = x
       /\ clock' = clock + 1
       /\ UNCHANGED <<orig, cur, cost>>

Unisolate ==
  /\ clock < MaxClock
  /\ \E x \in isol :
       /\ isol' = isol \ {x}
       /\ pick' = x
       /\ clock' = clock + 1
       /\ UNCHANGED <<orig, cur, cost>>

Done ==
  /\ \/ cost = MaxC
     \/ clock = MaxClock
  /\ UNCHANGED vars

Next == Query \/ Retest \/ Isolate \/ Unisolate \/ Done

Spec == Init /\ [][Next]_vars

IsolateSubsetOrig == isol \subseteq QueriedOrig

OrigIntegrity ==
  \A x \in X : orig[x] \in {None, Hidden[x]}

CurOnlyIfOrig ==
  \A x \in X : cur[x] # None => orig[x] # None

Safety ==
  /\ TypeOK
  /\ IsolateSubsetOrig
  /\ OrigIntegrity
  /\ CurOnlyIfOrig
====
