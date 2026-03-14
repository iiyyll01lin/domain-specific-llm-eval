"""
Main Pipeline Orchestrator for RAG Evaluation

Coordinates the entire evaluation pipeline from document processing
to final report generation.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.document_processor import DocumentProcessor
from data.hybrid_testset_generator import HybridTestsetGenerator
from distributed.federated_learning import FederatedLearningClient
from evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator
from evaluation.human_feedback_manager import HumanFeedbackManager
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.ragas_evaluator import RagasEvaluator
from interfaces.rag_interface import RAGInterface
from loaders.taxonomy_discovery import ZeroShotTaxonomyDiscoverer
from optimization.hyperparam_search import OptunaOptimizer
from pipeline.logger import (MemoryTracker, PerformanceTimer, log_stage_end,
                             log_stage_start)
from reports.report_generator import ReportGenerator
from ui.app_store_marketplace import UnifiedAppStore
from ui.force_graph_viewer import ForceGraphVisualizer
from utils.knowledge_graph_manager import KnowledgeGraphManager

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete RAG evaluation pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        run_id: str,
        output_dirs: Dict[str, Path],
        force_overwrite: bool = False,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration
            run_id: Unique run identifier
            output_dirs: Output directory structure
            force_overwrite: Whether to overwrite existing outputs
        """
        self.config = config
        self.run_id = run_id
        self.output_dirs = output_dirs
        self.force_overwrite = force_overwrite

        # Initialize components
        self.memory_tracker = MemoryTracker()

        # Initialize pipeline components
        self._initialize_components()

        logger.info(f"Pipeline orchestrator initialized for run: {run_id}")

    def run(self, stage: str = "all") -> Dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            stage: Pipeline stage to execute ('all', 'testset-generation', 'evaluation', 'reporting')

        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        results = {"success": False, "stages_completed": []}

        try:
            logger.info(f"🎬 Starting pipeline execution (stage: {stage})")
            self.memory_tracker.log_memory_usage("Pipeline Start")

            # Execute stages based on selection
            if stage in ["all", "testset-generation"]:
                results["testset_generation"] = self._run_testset_generation()
                results["stages_completed"].append("testset-generation")

                if not results["testset_generation"]["success"]:
                    raise Exception("Testset generation failed")

            if stage in ["all", "evaluation"]:
                results["evaluation"] = self._run_evaluation()
                results["stages_completed"].append("evaluation")

                if not results["evaluation"]["success"]:
                    raise Exception("Evaluation failed")

            if stage in ["all", "reporting"]:
                results["reporting"] = self._run_reporting()
                results["stages_completed"].append("reporting")

                if not results["reporting"]["success"]:
                    raise Exception("Reporting failed")

            # Pipeline completed successfully
            results["success"] = True
            duration = time.time() - start_time
            results["total_duration"] = duration

            logger.info(
                f"✅ Pipeline execution completed successfully in {duration:.2f} seconds"
            )
            self.memory_tracker.log_memory_usage("Pipeline End")

        except Exception as e:
            duration = time.time() - start_time
            results["error"] = str(e)
            results["total_duration"] = duration

            logger.error(
                f"❌ Pipeline execution failed after {duration:.2f} seconds: {e}"
            )
            self.memory_tracker.log_memory_usage("Pipeline Failed")

        return results

    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        logger.info("🏗️ Initializing pipeline components...")

        # Document processor (smart routing based on input type)
        self.document_processor = self._create_data_processor()

        # Testset generator (smart routing based on generator class)
        self.testset_generator = self._create_testset_generator()

        # RAG interface
        self.rag_interface = RAGInterface(config=self.config.get("rag_system", {}))

        # Evaluation components
        eval_config = self.config.get("evaluation", {})

        # Contextual keyword evaluator
        if eval_config.get("contextual_keywords", {}).get("enabled", False):
            self.contextual_evaluator = ContextualKeywordEvaluator(
                config=eval_config.get("contextual_keywords", {})
            )
        else:
            self.contextual_evaluator = None

        # RAGAS evaluator
        if eval_config.get("ragas_metrics", {}).get("enabled", False):
            self.ragas_evaluator = RagasEvaluator(
                config=eval_config.get("ragas_metrics", {})
            )
        else:
            self.ragas_evaluator = None

        # Human feedback manager
        if eval_config.get("human_feedback", {}).get("enabled", False):
            # Add output_dir to the human feedback config
            feedback_config = eval_config.get("human_feedback", {}).copy()
            feedback_config["output_dir"] = str(self.output_dirs["metadata"])
            self.feedback_manager = HumanFeedbackManager(
                config={"evaluation": {"human_feedback": feedback_config}}
            )
        else:
            self.feedback_manager = None

        # RAG evaluator (coordinator)
        self.rag_evaluator = RAGEvaluator(config=self.config)

        # Report generator
        self.report_generator = ReportGenerator(config=self.config)

        optimization_config = self.config.get("optimization", {}).get(
            "hyperparameter_search", {}
        )
        self.hyperparameter_optimizer = None
        if optimization_config.get("enabled", False):
            self.hyperparameter_optimizer = OptunaOptimizer(
                n_trials=int(optimization_config.get("n_trials", 6)),
                output_dir=str(self.output_dirs["metadata"] / "hyperparameter_search"),
            )

        taxonomy_config = self.config.get("taxonomy_discovery", {})
        self.taxonomy_discoverer = ZeroShotTaxonomyDiscoverer(
            llm_endpoint=str(taxonomy_config.get("endpoint", "local")),
        )
        self.topology_visualizer = ForceGraphVisualizer()

        app_store_config = self.config.get("app_store", {})
        self.app_store = UnifiedAppStore(
            manifest_dir=app_store_config.get("manifest_dir"),
            install_dir=self.output_dirs["metadata"] / "installed_runbooks",
        )

        federated_config = self.config.get("distributed", {}).get(
            "federated_learning", {}
        )
        self.federated_client = FederatedLearningClient(
            server_url=str(
                federated_config.get(
                    "server_url", "http://central-parameter-server.internal"
                )
            ),
            spool_dir=self.output_dirs["metadata"] / "federated_spool",
        )

        logger.info("✅ Pipeline components initialized")

    def _create_data_processor(self):
        """
        Create appropriate data processor based on input type.

        Returns:
            DocumentProcessor or CSVDataProcessor based on config
        """
        input_type = self.config.get("data_sources", {}).get("input_type", "documents")
        logger.info(f"🔍 Detected input type: {input_type}")

        if input_type == "csv":
            logger.info("📊 Creating CSV data processor...")
            from data.csv_data_processor import CSVDataProcessor

            return CSVDataProcessor(
                config=self.config.get("data_sources", {}),
                output_dir=self.output_dirs["temp"],
            )
        else:
            logger.info("📄 Creating document file processor...")
            return DocumentProcessor(
                config=self.config.get("custom_data", {}).get("data_sources", {}),
                output_dir=self.output_dirs["temp"],
            )

    def _create_testset_generator(self):
        """
        Create appropriate testset generator based on generator class.

        Returns:
            PureRagasTestsetGenerator or HybridTestsetGenerator based on config
        """
        generator_class = self.config.get("testset_generation", {}).get(
            "generator_class", "hybrid"
        )
        method = self.config.get("testset_generation", {}).get("method", "hybrid")

        logger.info(f"🔍 Detected generator class: {generator_class}")
        logger.info(f"🔍 Detected method: {method}")

        if generator_class == "pure_ragas_testset_generator" or method == "pure_ragas":
            logger.info("🧠 Creating Enhanced Pure RAGAS testset generator...")
            from data.pure_ragas_testset_generator import \
                PureRagasTestsetGenerator

            return PureRagasTestsetGenerator(
                config=self.config,  # Pass full config instead of just testset_generation section
                output_dir=self.output_dirs["testsets"],
            )
        else:
            logger.info("🔄 Creating Hybrid testset generator...")
            return HybridTestsetGenerator(config=self.config)

    def _run_hyperparameter_search(
        self, evaluation_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if self.hyperparameter_optimizer is None:
            return None
        return self.hyperparameter_optimizer.optimize(
            evaluator=lambda params: float(
                evaluation_results.get("success_rate", 0.0)
                + (params.get("metric_weight", 0.0) * 0.05)
            )
        )

    def _run_taxonomy_discovery(self) -> Optional[Dict[str, Any]]:
        taxonomy_config = self.config.get("taxonomy_discovery", {})
        if not taxonomy_config.get("enabled", False):
            return None
        csv_files = self.config.get("data_sources", {}).get("csv", {}).get("csv_files", [])
        if not csv_files:
            return None
        proposal_path = self.output_dirs["metadata"] / f"taxonomy_proposal_{self.run_id}.json"
        return self.taxonomy_discoverer.extract_ontology_from_csv_file(
            csv_files[0], persist_path=proposal_path
        )

    def _export_topology_artifact(self) -> Optional[Dict[str, str]]:
        topology_config = self.config.get("topology", {})
        if not topology_config.get("enabled", False):
            return None
        kg_manager = KnowledgeGraphManager(self.output_dirs["metadata"].parent)
        latest_kg = kg_manager.get_latest_knowledge_graph()
        if latest_kg is None:
            return None
        return self.topology_visualizer.export_from_kg_artifact(
            latest_kg, self.output_dirs["reports"] / "topology"
        )

    def _install_requested_runbooks(self) -> List[str]:
        app_store_config = self.config.get("app_store", {})
        if not app_store_config.get("enabled", False):
            return []
        installed: List[str] = []
        for runbook_id in app_store_config.get("auto_install", []):
            if self.app_store.install_runbook(str(runbook_id)):
                installed.append(str(runbook_id))
        return installed

    def _submit_federated_summary(
        self, evaluation_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        federated_config = self.config.get("distributed", {}).get(
            "federated_learning", {}
        )
        if not federated_config.get("enabled", False):
            return None
        return self.federated_client.submit_aggregation(
            [
                {
                    "node_id": self.run_id,
                    "score": float(evaluation_results.get("success_rate", 0.0)),
                    "sample_count": int(evaluation_results.get("total_queries", 0) or 1),
                    "tenant": str(federated_config.get("tenant", "default")),
                }
            ]
        )

    def _run_testset_generation(self) -> Dict[str, Any]:
        """
        Execute testset generation stage.

        Returns:
            Dictionary with stage results
        """
        log_stage_start(logger, "Testset Generation")
        stage_start = time.time()

        try:
            with PerformanceTimer("Testset Generation", logger):
                self.memory_tracker.log_memory_usage("Testset Generation Start")

                # Step 1: Process documents
                logger.info("📄 Processing documents...")
                processed_documents = self.document_processor.process_documents()

                if not processed_documents:
                    raise Exception("No documents were successfully processed")

                logger.info(f"✅ Processed {len(processed_documents)} documents")

                # Step 2: Generate testsets
                logger.info("🎯 Generating testsets...")

                # Extract document paths from processed documents
                document_paths = [doc["source_file"] for doc in processed_documents]

                # Use the comprehensive testset generation API
                testset_results = self.testset_generator.generate_comprehensive_testset(
                    document_paths=document_paths,
                    output_dir=self.output_dirs["testsets"],
                )

                # Extract results for compatibility
                testset_data = testset_results.get("testset", [])
                metadata_results = testset_results.get("metadata", {})
                generation_results = testset_results.get("results_by_method", {})

                logger.info(f"✅ Generated testset with {len(testset_data)} QA pairs")
                logger.info(
                    f"Generation method: {metadata_results.get('generation_method', 'unknown')}"
                )

                def make_json_serializable(obj):
                    """Convert objects to JSON-serializable format."""
                    import numpy as np
                    import pandas as pd

                    if hasattr(obj, "to_dict"):  # DataFrame
                        return obj.to_dict("records")
                    elif isinstance(obj, pd.Timestamp):  # Pandas Timestamp
                        return obj.isoformat()
                    elif isinstance(
                        obj, (pd.Series, np.ndarray)
                    ):  # Pandas Series or numpy array
                        return obj.tolist()
                    elif hasattr(obj, "isoformat"):  # datetime objects
                        return obj.isoformat()
                    elif hasattr(obj, "__dict__"):  # Custom objects
                        return {
                            k: make_json_serializable(v)
                            for k, v in obj.__dict__.items()
                        }
                    elif isinstance(obj, dict):
                        return {k: make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    else:
                        return obj

                # Step 3: Save metadata
                metadata = {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "documents_processed": len(processed_documents),
                    "testsets_generated": (
                        1
                        if (testset_data is not None and not testset_data.empty)
                        else 0
                    ),
                    "total_qa_pairs": (
                        len(testset_data) if testset_data is not None else 0
                    ),
                    "document_sources": [
                        doc["source_file"] for doc in processed_documents
                    ],
                    "generation_method": metadata_results.get(
                        "generation_method", "unknown"
                    ),
                    "generation_metadata": make_json_serializable(metadata_results),
                    "results_by_method": make_json_serializable(generation_results),
                }
                taxonomy_discovery = self._run_taxonomy_discovery()
                if taxonomy_discovery is not None:
                    metadata["taxonomy_discovery"] = make_json_serializable(
                        taxonomy_discovery
                    )
                installed_runbooks = self._install_requested_runbooks()
                if installed_runbooks:
                    metadata["installed_runbooks"] = installed_runbooks

                # Save metadata
                metadata_file = (
                    self.output_dirs["metadata"]
                    / f"testset_metadata_{self.run_id}.json"
                )
                import json

                # Make metadata JSON serializable
                serializable_metadata = make_json_serializable(metadata)
                with open(metadata_file, "w") as f:
                    json.dump(serializable_metadata, f, indent=2)

                self.memory_tracker.log_memory_usage("Testset Generation End")

                stage_duration = time.time() - stage_start
                log_stage_end(logger, "Testset Generation", True, stage_duration)

                return {
                    "success": True,
                    "documents_processed": len(processed_documents),
                    "testsets_generated": metadata["testsets_generated"],
                    "total_qa_pairs": metadata["total_qa_pairs"],
                    "generation_method": metadata["generation_method"],
                    "metadata_file": str(metadata_file),
                    "testset_data": testset_data,
                    "duration": stage_duration,
                }

        except Exception as e:
            stage_duration = time.time() - stage_start
            log_stage_end(logger, "Testset Generation", False, stage_duration)

            # Enhanced error logging for debugging
            import traceback

            logger.error(
                f"❌ Testset generation failed with {type(e).__name__}: {str(e)}"
            )
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": stage_duration,
            }

    def _run_evaluation(self) -> Dict[str, Any]:
        """
        Execute evaluation stage.

        Returns:
            Dictionary with stage results
        """
        log_stage_start(logger, "RAG Evaluation")
        stage_start = time.time()

        try:
            with PerformanceTimer("RAG Evaluation", logger):
                self.memory_tracker.log_memory_usage("Evaluation Start")

                # Step 1: Load testsets
                logger.info("📊 Loading testsets for evaluation...")
                testset_files = list(self.output_dirs["testsets"].glob("*.xlsx"))

                if not testset_files:
                    raise Exception("No testset files found for evaluation")

                # Step 2: Execute evaluation
                logger.info(
                    f"🔄 Evaluating RAG system with {len(testset_files)} testsets..."
                )
                evaluation_results = self.rag_evaluator.evaluate_testsets(testset_files)

                logger.info("✅ RAG evaluation completed")

                # Step 3: Save evaluation metadata
                metadata = {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "testsets_evaluated": len(testset_files),
                    "queries_executed": evaluation_results.get("total_queries", 0),
                    "keyword_pass_rate": evaluation_results.get(
                        "keyword_metrics", {}
                    ).get("pass_rate", 0),
                    "avg_ragas_score": evaluation_results.get("ragas_metrics", {}).get(
                        "average_score", 0
                    ),
                    "feedback_requests": evaluation_results.get(
                        "feedback_metrics", {}
                    ).get("requests", 0),
                    "evaluation_file": evaluation_results.get("output_file", ""),
                }
                hyperparameter_search = self._run_hyperparameter_search(
                    evaluation_results
                )
                if hyperparameter_search is not None:
                    metadata["hyperparameter_search"] = hyperparameter_search
                federated_submission = self._submit_federated_summary(
                    evaluation_results
                )
                if federated_submission is not None:
                    metadata["federated_submission"] = federated_submission

                # Save evaluation metadata
                eval_metadata_file = (
                    self.output_dirs["metadata"]
                    / f"evaluation_metadata_{self.run_id}.json"
                )
                import json

                with open(eval_metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                self.memory_tracker.log_memory_usage("Evaluation End")

                stage_duration = time.time() - stage_start
                log_stage_end(logger, "RAG Evaluation", True, stage_duration)

                return {
                    "success": True,
                    "testsets_evaluated": len(testset_files),
                    "queries_executed": metadata["queries_executed"],
                    "keyword_pass_rate": metadata["keyword_pass_rate"],
                    "avg_ragas_score": metadata["avg_ragas_score"],
                    "feedback_requests": metadata["feedback_requests"],
                    "evaluation_file": metadata["evaluation_file"],
                    "metadata_file": str(eval_metadata_file),
                    "duration": stage_duration,
                }

        except Exception as e:
            stage_duration = time.time() - stage_start
            log_stage_end(logger, "RAG Evaluation", False, stage_duration)

            return {"success": False, "error": str(e), "duration": stage_duration}

    def _run_reporting(self) -> Dict[str, Any]:
        """
        Execute reporting stage.

        Returns:
            Dictionary with stage results
        """
        log_stage_start(logger, "Report Generation")
        stage_start = time.time()

        try:
            with PerformanceTimer("Report Generation", logger):
                self.memory_tracker.log_memory_usage("Reporting Start")

                # Step 1: Load evaluation data
                logger.info("📈 Loading evaluation data for reporting...")

                # Find evaluation results - check both metadata and evaluations directories
                eval_files = list(
                    self.output_dirs["metadata"].glob("*evaluation_results*.json")
                )
                if not eval_files:
                    # Also check evaluations directory
                    eval_files = list(
                        self.output_dirs["evaluations"].glob(
                            "*evaluation_results*.json"
                        )
                    )
                if not eval_files:
                    raise Exception("No evaluation results found for reporting")

                # Step 2: Generate reports
                logger.info("📝 Generating evaluation reports...")
                report_results = self.report_generator.generate_reports(
                    evaluation_files=eval_files, run_id=self.run_id
                )

                logger.info(f"✅ Generated {len(report_results)} reports")

                # Step 3: Save reporting metadata
                metadata = {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "reports_generated": len(report_results),
                    "report_types": [r["type"] for r in report_results],
                    "report_files": [r["file_path"] for r in report_results],
                    "report_directory": str(self.output_dirs["reports"]),
                }
                topology_artifact = self._export_topology_artifact()
                if topology_artifact is not None:
                    metadata["topology_artifact"] = topology_artifact

                # Save reporting metadata
                report_metadata_file = (
                    self.output_dirs["metadata"] / f"report_metadata_{self.run_id}.json"
                )
                import json

                with open(report_metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                self.memory_tracker.log_memory_usage("Reporting End")

                stage_duration = time.time() - stage_start
                log_stage_end(logger, "Report Generation", True, stage_duration)

                return {
                    "success": True,
                    "reports_generated": report_results,
                    "report_directory": str(self.output_dirs["reports"]),
                    "metadata_file": str(report_metadata_file),
                    "duration": stage_duration,
                }

        except Exception as e:
            stage_duration = time.time() - stage_start
            log_stage_end(logger, "Report Generation", False, stage_duration)

            return {"success": False, "error": str(e), "duration": stage_duration}
