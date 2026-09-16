------------------------------- MODULE TASK -------------------------------
EXTENDS Integers, Sequences, FiniteSets

\* Task definition only: study the research process, not one domain solution.
\* No previous project, historical finding or preferred framework is supplied.
ResearchObject == "ResearchPolicy"
EvaluationUnit == "ResearchEpisode"
Question == "HowToDiscoverAndReviseRepresentationsObjectivesAndAlgorithmsByExperiments"
Unknowns == {"Representation", "OptimizationObjective", "EffectiveAlgorithm"}
ExternalPurposeMayChangeToManufactureSuccess == FALSE

InitialProjectData == {}
InheritedConversation == <<>>
InheritedFindings == {}
PreselectedFramework == {}
FixedHypothesisLibraryRequired == FALSE

AllowedInputs == {"CurrentTaskDefinition", "RelevantPublicPrimaryLiterature",
                  "NewlyGeneratedExperiments", "CurrentExecutionEnvironment"}
PermittedResearchActions ==
    {"GenerateHypothesis", "DesignExperiment", "Observe",
     "ReviseRepresentation", "ReviseOptimizationObjective",
     "ConstructAlgorithm", "SelectNextAction", "CompareResearchPolicies"}

\* Record predicates express what must be evidenced, not self-certification.
FormulationOK(f) ==
    /\ f.object = ResearchObject /\ f.unit = EvaluationUnit
    /\ f.unknownsExplicit /\ f.informationAndActionsExplicit
    /\ f.allowsNewHypothesesRepresentationsAndExperiments
    /\ f.distinguishesExternalPurposeFromOptimizedProxy
    /\ f.equivalenceClaimHasStatedScope

ComparisonOK(c) ==
    /\ c.comparesResearchPolicies
    /\ c.sameInitialTaskInformation /\ c.sameResourceConstraints
    /\ c.sameExternallyFixedEvaluation
    /\ c.recordsResearchTrajectoryAndTerminalResult
    /\ c.parametersHaveProvenance /\ c.assumptionsLabelled
    /\ c.negativeResultsRetained

ContextAblationOK(a) ==
    /\ a.sameTaskInstructions /\ a.sameModelAndToolConfiguration
    /\ a.sameBudgetAndEvaluation /\ a.newSessionPerRun
    /\ a.onlyManipulatedFactor = "PriorProjectContext"

EvidenceBoundaryOK(r) ==
    /\ ~r.callsToySuccessEstablishedGeneralResearchImprovement
    /\ ~r.callsOneDomainAlgorithmGainResearchPolicyGainWithoutComparison
    /\ ~r.callsPreparingCleanWorkspaceCompletedAblation

Deliverables == {"FORMULATION.md", "METHOD.md", "experiment.py", "RESULTS.json"}
DeliveryOK(d) ==
    /\ d.requiredFiles = Deliverables
    /\ d.formulationProvided /\ d.methodExplained
    /\ d.executableComparisonProvided /\ d.actualResultsReported
    /\ d.executedAndUnexecutedWorkDistinguished

SessionContract(r) ==
    /\ r.initialProjectData = InitialProjectData
    /\ r.inheritedConversation = InheritedConversation
    /\ r.inheritedFindings = InheritedFindings
    /\ r.inputKinds \subseteq AllowedInputs
    /\ ~r.readsOtherLocalProjects /\ ~r.usesPriorProjectSpecificSkills
    /\ r.writesOnlyInsideCurrentWorkspace
    /\ FormulationOK(r.formulation)
    /\ ComparisonOK(r.comparison)
    /\ EvidenceBoundaryOK(r)
    /\ (r.claimsCompletedDelivery => DeliveryOK(r.delivery))
    /\ (r.claimsContextAblation => ContextAblationOK(r.ablation))

\* New-session isolation must be supplied by the host. This specification
\* neither clears an existing conversation nor enforces filesystem isolation.
=============================================================================
