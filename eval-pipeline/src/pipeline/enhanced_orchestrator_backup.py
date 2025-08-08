"""
Enhanced Pipeline Orchestrator with Comprehensive Validation

This orchestrator integrates all validation systems to ensure robust pipeline execution:
- CSV data validation and cleaning
- Sample validation during testset generation  
- Knowledge graph validation
- Comprehensive error handling and recovery

Phase 2: Architecture Cleanup Implementation
This version now uses the refactored enhanced orchestrator with separated concerns.
"""

from pipeline.refactored_enhanced_orchestrator import RefactoredEnhancedPipelineOrchestrator

# For backward compatibility, alias the refactored version
class EnhancedPipelineOrchestrator(RefactoredEnhancedPipelineOrchestrator):
    """
    Enhanced pipeline orchestrator with comprehensive validation and error recovery.
    
    Phase 2: This class now uses the refactored architecture with separated
    core execution logic and validation/enhancement features.
    """
    pass
        
        # Initialize robust sample processor
        self.robust_processor = RobustSampleProcessor(
            min_success_rate=validation_config.get('min_success_rate', 0.02),
            enable_aggressive_recovery=validation_config.get('enable_aggressive_recovery', True),
            enable_drop_and_continue=validation_config.get('enable_drop_and_continue', True),
            min_content_length=validation_config.get('min_content_length', 5)
        )
        
        self.kg_validator = KnowledgeGraphValidator(config)
        
        # Initialize validation tracking
        self.validation_reports = {
            'csv_validation': [],
            'sample_validation': [],
            'kg_validation': [],
            'overall_stats': {}
        }
        
        # Call parent constructor
        super().__init__(config, run_id, output_dirs, force_overwrite)
        
        logger.info("🛡️ Enhanced pipeline orchestrator initialized with robust validation")
        logger.info(f"   Sample success rate threshold: {self.sample_validator.min_success_rate:.1%}")
        logger.info(f"   Robust processing enabled: {validation_config.get('enable_drop_and_continue', True)}")
        logger.info(f"   Aggressive recovery enabled: {validation_config.get('enable_aggressive_recovery', True)}")
        logger.info(f"   Min content length: {validation_config.get('min_content_length', 5)}")
    
    def run(self, stage: str = "all") -> Dict[str, Any]:
        """
        Execute the enhanced pipeline with comprehensive validation.
        
        Args:
            stage: Pipeline stage to execute
            
        Returns:
            Dictionary with execution results including validation reports
        """
        start_time = time.time()
        results = {'success': False, 'stages_completed': [], 'validation_reports': {}}
        
        try:
            logger.info(f"🎬 Starting enhanced pipeline execution (stage: {stage})")
            logger.info("🛡️ Pre-execution validation checks...")
            
            # Pre-execution validation
            pre_validation_results = self._run_pre_execution_validation()
            results['validation_reports']['pre_execution'] = pre_validation_results
            
            if not pre_validation_results['success']:
                # Don't fail immediately - try to recover
                logger.warning("⚠️ Pre-execution validation found issues, attempting recovery...")
                recovery_results = self._attempt_data_recovery(pre_validation_results)
                results['validation_reports']['recovery'] = recovery_results
                
                if not recovery_results['success']:
                    raise Exception(f"Pre-execution validation failed and recovery unsuccessful: {recovery_results['error']}")
                else:
                    logger.info("✅ Data recovery successful, continuing with pipeline")
            
            # Execute stages with enhanced validation
            if stage in ["all", "testset-generation"]:
                results['testset_generation'] = self._run_enhanced_testset_generation()
                results['stages_completed'].append('testset-generation')
                
                if not results['testset_generation']['success']:
                    # Try recovery before failing
                    recovery_result = self._recover_testset_generation(results['testset_generation'])
                    if recovery_result['success']:
                        results['testset_generation'] = recovery_result
                        logger.info("✅ Testset generation recovered successfully")
                    else:
                        raise Exception("Testset generation failed and could not be recovered")
            
            # ❌ COMMENTED OUT: Unimplemented keyword extraction stage
            # This stage is not implemented and causes different execution paths between
            # --stage testset-generation and --stage all, leading to non-deterministic results.
            # Keywords are already generated within testset generation via UnifiedKeyBERTExtractor.
            #
            # if stage in ["all", "keyword-extraction"]:
            #     results['keyword_extraction'] = self._run_keyword_extraction()
            #     results['stages_completed'].append('keyword-extraction')
            #     
            #     if not results['keyword_extraction']['success']:
            #         raise Exception("Keyword extraction failed")
            
            if stage in ["all", "evaluation"]:
                results['evaluation'] = self._run_evaluation()
                results['stages_completed'].append('evaluation')
                
                if not results['evaluation']['success']:
                    raise Exception("Evaluation failed")
            
            if stage in ["all", "reporting"]:
                results['reporting'] = self._run_reporting()
                results['stages_completed'].append('reporting')
                
                if not results['reporting']['success']:
                    raise Exception("Reporting failed")
            
            # Post-execution validation
            post_validation_results = self._run_post_execution_validation(results)
            results['validation_reports']['post_execution'] = post_validation_results
            
            # Generate comprehensive validation report
            validation_summary = self._generate_validation_summary()
            results['validation_reports']['summary'] = validation_summary
            
            # Save validation reports
            validation_report_path = self._save_validation_reports(results['validation_reports'])
            results['validation_report_file'] = str(validation_report_path)
            
            # Pipeline completed successfully
            results['success'] = True
            duration = time.time() - start_time
            results['total_duration'] = duration
            
            logger.info(f"✅ Enhanced pipeline execution completed successfully in {duration:.2f} seconds")
            logger.info(f"📊 Validation summary: {validation_summary['overall_quality_score']:.1%} data quality")
            
        except Exception as e:
            duration = time.time() - start_time
            results['error'] = str(e)
            results['total_duration'] = duration
            
            # Generate error analysis
            error_analysis = self._analyze_pipeline_error(e, results)
            results['error_analysis'] = error_analysis
            
            logger.error(f"❌ Enhanced pipeline execution failed after {duration:.2f} seconds: {e}")
            
            # Save partial results and validation reports for debugging
            try:
                validation_report_path = self._save_validation_reports(results.get('validation_reports', {}))
                results['partial_validation_report_file'] = str(validation_report_path)
            except Exception as save_error:
                logger.error(f"Failed to save validation reports: {save_error}")
        
        return results
    
    def _run_pre_execution_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive pre-execution validation.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Running pre-execution validation...")
        
        validation_results = {
            'success': True,
            'csv_validation': None,
            'kg_validation': None,
            'config_validation': None,
            'issues_found': [],
            'fixes_applied': [],
            'recommendations': []
        }
        
        try:
            # Validate CSV data if CSV input is configured
            if self.config.get('input', {}).get('type') == 'csv':
                csv_files = self.config.get('input', {}).get('csv_files', [])
                if csv_files:
                    logger.info(f"📊 Validating {len(csv_files)} CSV files...")
                    csv_results = self._validate_csv_files(csv_files)
                    validation_results['csv_validation'] = csv_results
                    
                    if not csv_results['success']:
                        validation_results['success'] = False
                        validation_results['issues_found'].extend(csv_results['issues'])
            
            # Validate knowledge graph if available
            kg_files = self._find_knowledge_graph_files()
            if kg_files:
                logger.info(f"🧠 Validating {len(kg_files)} knowledge graph files...")
                kg_results = self._validate_kg_files(kg_files)
                validation_results['kg_validation'] = kg_results
                
                if not kg_results['success']:
                    validation_results['issues_found'].extend(kg_results['issues'])
            
            # Validate configuration
            config_results = self._validate_pipeline_config()
            validation_results['config_validation'] = config_results
            
            if not config_results['success']:
                validation_results['success'] = False
                validation_results['issues_found'].extend(config_results['issues'])
            
            logger.info(f"✅ Pre-execution validation completed: {len(validation_results['issues_found'])} issues found")
            
        except Exception as e:
            validation_results['success'] = False
            validation_results['error'] = str(e)
            logger.error(f"❌ Pre-execution validation failed: {e}")
        
        return validation_results
    
    def _validate_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """Validate and potentially clean CSV files."""
        
        results = {
            'success': True,
            'files_processed': 0,
            'files_cleaned': 0,
            'total_issues': 0,
            'total_fixes': 0,
            'issues': [],
            'cleaned_files': [],
            'validation_reports': []
        }
        
        try:
            for csv_file_path in csv_files:
                csv_file = Path(csv_file_path)
                if not csv_file.exists():
                    results['issues'].append(f"CSV file not found: {csv_file}")
                    results['success'] = False
                    continue
                
                # Validate and clean CSV
                logger.info(f"🔍 Validating CSV file: {csv_file.name}")
                cleaned_df, validation_report = self.csv_validator.validate_and_clean_csv(csv_file)
                
                results['files_processed'] += 1
                results['validation_reports'].append(validation_report)
                results['total_issues'] += len(validation_report['issues_found'])
                results['total_fixes'] += len(validation_report['fixes_applied'])
                
                # Check if cleaning was needed
                if validation_report['fixes_applied']:
                    # Save cleaned CSV
                    output_dir = self.output_dirs['base'] / "preprocessed_data"
                    cleaned_csv_path = self.csv_validator.save_cleaned_csv(cleaned_df, csv_file, output_dir)
                    results['cleaned_files'].append(str(cleaned_csv_path))
                    results['files_cleaned'] += 1
                    
                    # Update config to use cleaned file
                    self._update_config_with_cleaned_file(csv_file_path, str(cleaned_csv_path))
                
                # Check data quality
                if validation_report['data_quality_score'] < 0.5:
                    results['issues'].append(f"Low data quality in {csv_file.name}: {validation_report['data_quality_score']:.1%}")
                    results['success'] = False
                
                logger.info(f"✅ CSV validation complete for {csv_file.name}: {validation_report['data_quality_score']:.1%} quality")
        
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"❌ CSV validation failed: {e}")
        
        # Save validation summary with error handling
        if results['validation_reports']:
            try:
                report_dir = self.output_dirs['base'] / "validation_reports"
                report_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                self.csv_validator.save_validation_report(
                    results['validation_reports'],
                    report_dir
                )
                logger.info("📄 Saved CSV validation report")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save CSV validation report: {e}")
                # Don't fail the entire validation for this
        
        return results
    
    def _validate_kg_files(self, kg_files: List[Path]) -> Dict[str, Any]:
        """Validate knowledge graph files."""
        
        results = {
            'success': True,
            'files_processed': 0,
            'files_cleaned': 0,
            'issues': [],
            'cleaned_files': [],
            'validation_reports': []
        }
        
        try:
            for kg_file in kg_files:
                logger.info(f"🧠 Validating KG file: {kg_file.name}")
                cleaned_kg, validation_report = self.kg_validator.validate_kg_file(kg_file)
                
                results['files_processed'] += 1
                results['validation_reports'].append(validation_report)
                
                # Check if cleaning was needed
                if validation_report['fixes_applied']:
                    # Save cleaned KG
                    output_dir = self.output_dirs['base'] / "preprocessed_data"
                    cleaned_kg_path = self.kg_validator.save_cleaned_kg(cleaned_kg, kg_file, output_dir)
                    results['cleaned_files'].append(str(cleaned_kg_path))
                    results['files_cleaned'] += 1
                
                # Check integrity
                integrity_results = KGIntegrityChecker.check_graph_connectivity(cleaned_kg)
                if integrity_results.get('is_connected', False):
                    logger.info(f"✅ KG {kg_file.name} is well-connected")
                else:
                    results['issues'].append(f"KG {kg_file.name} has connectivity issues: {integrity_results.get('connected_components', 0)} components")
                
                logger.info(f"✅ KG validation complete for {kg_file.name}")
        
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"❌ KG validation failed: {e}")
        
        # Save validation summary with error handling
        if results['validation_reports']:
            try:
                report_dir = self.output_dirs['base'] / "validation_reports"
                report_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                self.kg_validator.save_validation_report(
                    results['validation_reports'],
                    report_dir
                )
                logger.info("📄 Saved KG validation report")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save KG validation report: {e}")
                # Don't fail the entire validation for this
        
        return results
    
    def _validate_pipeline_config(self) -> Dict[str, Any]:
        """Validate pipeline configuration for common issues."""
        
        results = {
            'success': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check required sections - allow flexible naming
            # Accept either 'input' or 'data_sources' for input configuration
            input_section_found = 'input' in self.config or 'data_sources' in self.config
            if not input_section_found:
                results['issues'].append("Missing required config section: input/data_sources")
                results['success'] = False
            
            # Check other required sections
            required_sections = ['testset_generation', 'evaluation', 'output']
            for section in required_sections:
                if section not in self.config:
                    results['issues'].append(f"Missing required config section: {section}")
                    results['success'] = False
            
            # Check input configuration - handle both 'input' and 'data_sources' structures
            input_config = self.config.get('input', {})
            data_sources_config = self.config.get('data_sources', {})
            
            # Check for CSV configuration in either structure
            csv_files = []
            if input_config.get('type') == 'csv':
                csv_files = input_config.get('csv_files', [])
            elif 'csv' in data_sources_config:
                csv_files = data_sources_config.get('csv', {}).get('csv_files', [])
            
            # If CSV configuration is present but no files specified
            if (input_config.get('type') == 'csv' or 'csv' in data_sources_config) and not csv_files:
                results['issues'].append("CSV configuration found but no csv_files provided")
                results['success'] = False
            
            # Check testset generation configuration
            testset_config = self.config.get('testset_generation', {})
            target_samples = testset_config.get('max_total_samples', 0)
            if target_samples > 10000:
                results['warnings'].append(f"Large testset target ({target_samples}) may take significant time")
                results['recommendations'].append("Consider using batch processing for large testsets")
            
            # Check LLM configuration
            custom_llm = testset_config.get('ragas_config', {}).get('custom_llm', {})
            if not custom_llm.get('api_key'):
                results['warnings'].append("No custom LLM API key found - using default OpenAI")
            
            # Check validation settings
            validation_config = self.config.get('validation', {})
            if not validation_config:
                results['recommendations'].append("Add validation configuration for better error handling")
            
            logger.info(f"✅ Configuration validation: {len(results['issues'])} issues, {len(results['warnings'])} warnings")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"❌ Configuration validation failed: {e}")
        
        return results
    
    def _attempt_data_recovery(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from data validation issues."""
        
        recovery_results = {
            'success': False,
            'actions_taken': [],
            'issues_resolved': 0,
            'issues_remaining': 0,
            'error': None
        }
        
        try:
            logger.info("🔧 Attempting data recovery...")
            
            # Recover from CSV issues
            csv_validation = validation_results.get('csv_validation')
            if csv_validation and not csv_validation['success']:
                csv_recovery = self._recover_csv_issues(csv_validation)
                recovery_results['actions_taken'].extend(csv_recovery['actions'])
                if csv_recovery['success']:
                    recovery_results['issues_resolved'] += csv_recovery['issues_resolved']
            
            # Recover from KG issues  
            kg_validation = validation_results.get('kg_validation')
            if kg_validation and not kg_validation['success']:
                kg_recovery = self._recover_kg_issues(kg_validation)
                recovery_results['actions_taken'].extend(kg_recovery['actions'])
                if kg_recovery['success']:
                    recovery_results['issues_resolved'] += kg_recovery['issues_resolved']
            
            # Check if recovery was successful
            total_issues = len(validation_results.get('issues_found', []))
            recovery_results['issues_remaining'] = total_issues - recovery_results['issues_resolved']
            
            if recovery_results['issues_remaining'] == 0:
                recovery_results['success'] = True
                logger.info(f"✅ Data recovery successful: {recovery_results['issues_resolved']} issues resolved")
            else:
                logger.warning(f"⚠️ Partial recovery: {recovery_results['issues_remaining']} issues remain")
        
        except Exception as e:
            recovery_results['error'] = str(e)
            logger.error(f"❌ Data recovery failed: {e}")
        
        return recovery_results
    
    def _recover_csv_issues(self, csv_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from CSV validation issues."""
        
        recovery = {
            'success': False,
            'actions': [],
            'issues_resolved': 0
        }
        
        try:
            # If we have cleaned files, update the configuration
            if csv_validation.get('cleaned_files'):
                original_files = self.config.get('input', {}).get('csv_files', [])
                cleaned_files = csv_validation['cleaned_files']
                
                # Update config to use cleaned files
                self.config['input']['csv_files'] = cleaned_files
                recovery['actions'].append(f"Updated config to use {len(cleaned_files)} cleaned CSV files")
                recovery['issues_resolved'] += len(cleaned_files)
                recovery['success'] = True
                
                logger.info(f"🔧 Updated configuration to use cleaned CSV files")
        
        except Exception as e:
            logger.error(f"❌ CSV recovery failed: {e}")
        
        return recovery
    
    def _recover_kg_issues(self, kg_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from knowledge graph validation issues."""
        
        recovery = {
            'success': False,
            'actions': [],
            'issues_resolved': 0
        }
        
        try:
            # If we have cleaned KG files, update relevant configurations
            if kg_validation.get('cleaned_files'):
                cleaned_files = kg_validation['cleaned_files']
                
                # Update KG configuration if it exists
                kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
                if kg_config and len(cleaned_files) > 0:
                    kg_config['existing_kg_file'] = cleaned_files[0]  # Use first cleaned file
                    recovery['actions'].append(f"Updated KG config to use cleaned file: {cleaned_files[0]}")
                    recovery['issues_resolved'] += 1
                    recovery['success'] = True
                
                logger.info(f"🔧 Updated KG configuration to use cleaned files")
        
        except Exception as e:
            logger.error(f"❌ KG recovery failed: {e}")
        
        return recovery
    
    def _run_enhanced_testset_generation(self) -> Dict[str, Any]:
        """
        Run testset generation with ultra-robust sample validation.
        
        Returns:
            Dictionary with testset generation results
        """
        logger.info("🚀 Starting enhanced testset generation with robust sample processing...")
        start_time = time.time()
        
        try:
            # Run the original testset generation
            base_results = super()._run_testset_generation()
            
            if not base_results['success']:
                return base_results
            
            # Enhanced robust validation of generated samples
            if 'testset_data' in base_results:
                samples = base_results['testset_data']
                logger.info(f"🔍 Robustly processing {len(samples)} generated samples...")
                
                # Use robust processor instead of standard validator
                valid_samples, processing_report = self.robust_processor.process_samples_robustly(samples)
                
                # Store processing report
                self.validation_reports['sample_validation'].append(processing_report)
                
                # Update results with processed samples
                base_results['testset_data'] = valid_samples
                base_results['samples_validated'] = processing_report['valid_count']
                base_results['samples_recovered'] = processing_report['recovered_count']
                base_results['samples_dropped'] = processing_report['dropped_count']
                base_results['validation_success_rate'] = processing_report['success_rate']
                base_results['robust_processing_enabled'] = True
                
                logger.info(f"✅ Robust processing complete:")
                logger.info(f"   Valid: {len(valid_samples)}/{len(samples)} samples ({processing_report['success_rate']:.1%})")
                logger.info(f"   Recovered: {processing_report['recovered_count']} samples")
                logger.info(f"   Dropped: {processing_report['dropped_count']} samples")
                
                # Only trigger recovery if success rate is extremely low (< 2%)
                if processing_report.get('pipeline_recovery_recommended', False):
                    logger.warning("⚠️ Very low success rate detected, attempting additional recovery...")
                    
                    # Generate additional samples only if critically needed
                    recovery_samples = self._generate_recovery_samples(
                        target_count=len(samples),
                        current_count=len(valid_samples)
                    )
                    
                    if recovery_samples:
                        # Process recovery samples robustly
                        validated_recovery, recovery_processing = self.robust_processor.process_samples_robustly(recovery_samples)
                        
                        # Add valid recovery samples
                        valid_samples.extend(validated_recovery)
                        base_results['testset_data'] = valid_samples
                        base_results['recovery_samples_generated'] = len(recovery_samples)
                        base_results['recovery_samples_valid'] = len(validated_recovery)
                        
                        logger.info(f"🔧 Additional recovery generated {len(validated_recovery)} valid samples")
                else:
                    logger.info("✅ Success rate acceptable - no additional recovery needed")
                
                # Save validated samples to file
                if valid_samples:
                    validated_testset_path = self._save_validated_testset(valid_samples)
                    base_results['validated_testset_path'] = str(validated_testset_path)
                    
                    # Save processing statistics
                    stats = self.robust_processor.get_processing_statistics()
                    base_results['processing_statistics'] = stats
                    
                    logger.info(f"📊 Overall retention rate: {stats['overall_retention_rate']:.1%}")
                
                # Save processing report
                processing_report_path = self._save_processing_report(processing_report)
                base_results['processing_report'] = str(processing_report_path)
            
            base_results['duration'] = time.time() - start_time
            return base_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced testset generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def _generate_recovery_samples(self, target_count: int, current_count: int) -> List[Dict[str, Any]]:
        """Generate additional samples to recover from validation losses."""
        
        needed_count = target_count - current_count
        if needed_count <= 0:
            return []
        
        logger.info(f"🔧 Generating {needed_count} recovery samples...")
        
        try:
            # Use a simplified generation approach for recovery
            recovery_samples = []
            
            # Generate simple question-answer pairs based on configuration
            csv_files = self.config.get('input', {}).get('csv_files', [])
            if csv_files:
                # Load some data from CSV for sample generation
                try:
                    import pandas as pd
                    csv_file = Path(csv_files[0])
                    if csv_file.exists():
                        df = pd.read_csv(csv_file)
                        
                        # Generate simple samples from CSV data
                        for i in range(min(needed_count, len(df))):
                            row = df.iloc[i]
                            
                            # Create a basic sample structure
                            sample = {
                                'user_input': f"What is the solution for {row.get('display', 'this issue')}?",
                                'reference': f"The solution involves {row.get('content', 'standard procedures')}.",
                                'reference_contexts': [str(row.get('display', 'context'))],
                                'eval_sample': {
                                    'user_input': f"What is the solution for {row.get('display', 'this issue')}?",
                                    'reference': f"The solution involves {row.get('content', 'standard procedures')}.",
                                    'reference_contexts': [str(row.get('display', 'context'))]
                                },
                                'synthesizer_name': 'recovery_generator'
                            }
                            
                            recovery_samples.append(sample)
                            
                except Exception as e:
                    logger.warning(f"⚠️ CSV-based recovery generation failed: {e}")
            
            logger.info(f"✅ Generated {len(recovery_samples)} recovery samples")
            return recovery_samples
            
        except Exception as e:
            logger.error(f"❌ Recovery sample generation failed: {e}")
            return []
    
    def _save_validated_testset(self, valid_samples: List[Dict[str, Any]]) -> Path:
        """Save validated samples to a testset file."""
        
        try:
            import pandas as pd
            
            # Convert samples to DataFrame
            df = pd.DataFrame(valid_samples)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validated_testset_{self.run_id}_{timestamp}.csv"
            output_path = self.output_dirs['testsets'] / filename
            
            df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved validated testset: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save validated testset: {e}")
            # Fallback to JSON
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validated_testset_{self.run_id}_{timestamp}.json"
            output_path = self.output_dirs['testsets'] / filename
            
            with open(output_path, 'w') as f:
                json.dump(valid_samples, f, indent=2)
            
            return output_path
    
    def _save_processing_report(self, processing_report: Dict[str, Any]) -> Path:
        """Save robust processing report."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"robust_processing_report_{self.run_id}_{timestamp}.json"
            report_path = self.output_dirs['base'] / "validation_reports" / report_filename
            
            # Ensure directory exists
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            comprehensive_report = {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'processing_report': processing_report,
                'processor_statistics': self.robust_processor.get_processing_statistics(),
                'configuration': {
                    'min_success_rate': self.robust_processor.min_success_rate,
                    'aggressive_recovery': self.robust_processor.enable_aggressive_recovery,
                    'drop_and_continue': self.robust_processor.enable_drop_and_continue,
                    'min_content_length': self.robust_processor.min_content_length
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Saved robust processing report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save processing report: {e}")
            return self.output_dirs['base'] / "processing_report_save_failed.txt"
    
    def _recover_testset_generation(self, failed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from testset generation failure."""
        
        logger.info("🔧 Attempting testset generation recovery...")
        
        recovery_results = {
            'success': False,
            'recovery_method': None,
            'samples_recovered': 0,
            'error': None
        }
        
        try:
            # Try different recovery strategies
            
            # Strategy 1: Use smaller batch size
            if 'samples_generated' in failed_results and failed_results.get('samples_generated', 0) > 0:
                logger.info("🔄 Strategy 1: Reduce batch size and retry")
                
                # Reduce target size
                original_target = self.config.get('testset_generation', {}).get('max_total_samples', 10)
                reduced_target = max(3, original_target // 10)  # Reduce to 10% or minimum 3
                
                # Update config temporarily
                original_config = self.config['testset_generation'].copy()
                self.config['testset_generation']['max_total_samples'] = reduced_target
                
                try:
                    # Retry generation with reduced parameters
                    retry_results = super()._run_testset_generation()
                    
                    if retry_results['success']:
                        recovery_results['success'] = True
                        recovery_results['recovery_method'] = 'reduced_batch_size'
                        recovery_results['samples_recovered'] = retry_results.get('total_qa_pairs', 0)
                        recovery_results.update(retry_results)  # Include all retry results
                        
                        logger.info(f"✅ Recovery successful: generated {recovery_results['samples_recovered']} samples with reduced target")
                        return recovery_results
                        
                finally:
                    # Restore original config
                    self.config['testset_generation'] = original_config
            
            # Strategy 2: Generate minimal testset using fallback method
            logger.info("🔄 Strategy 2: Generate minimal fallback testset")
            minimal_samples = self._generate_minimal_testset()
            
            if minimal_samples:
                # Validate minimal samples
                valid_samples, validation_report = self.sample_validator.validate_batch(minimal_samples)
                
                if valid_samples:
                    # Save minimal testset
                    testset_path = self._save_validated_testset(valid_samples)
                    
                    recovery_results['success'] = True
                    recovery_results['recovery_method'] = 'minimal_fallback'
                    recovery_results['samples_recovered'] = len(valid_samples)
                    recovery_results['testset_data'] = valid_samples
                    recovery_results['testset_path'] = str(testset_path)
                    recovery_results['validation_success_rate'] = validation_report['success_rate']
                    
                    logger.info(f"✅ Fallback recovery successful: generated {len(valid_samples)} minimal samples")
                    return recovery_results
            
            # If all strategies fail
            recovery_results['error'] = "All recovery strategies failed"
            logger.error("❌ All testset generation recovery strategies failed")
            
        except Exception as e:
            recovery_results['error'] = str(e)
            logger.error(f"❌ Testset generation recovery failed: {e}")
        
        return recovery_results
    
    def _generate_minimal_testset(self) -> List[Dict[str, Any]]:
        """Generate a minimal testset for fallback scenarios."""
        
        minimal_samples = []
        
        try:
            # Strategy: Use CSV data to create basic question-answer pairs
            csv_files = self.config.get('input', {}).get('csv_files', [])
            
            if csv_files:
                import pandas as pd
                
                for csv_file_path in csv_files[:1]:  # Use only first CSV file
                    csv_file = Path(csv_file_path)
                    if csv_file.exists():
                        df = pd.read_csv(csv_file)
                        
                        # Take first few rows to create samples
                        for i, row in df.head(3).iterrows():
                            display = str(row.get('display', f'Item {i}'))
                            content = str(row.get('content', 'No content available'))
                            
                            sample = {
                                'user_input': f"What can you tell me about {display}?",
                                'reference': f"Based on the information available: {content}",
                                'reference_contexts': [display],
                                'eval_sample': {
                                    'user_input': f"What can you tell me about {display}?",
                                    'reference': f"Based on the information available: {content}",
                                    'reference_contexts': [display]
                                },
                                'synthesizer_name': 'minimal_fallback_generator'
                            }
                            
                            minimal_samples.append(sample)
                            
                        logger.info(f"✅ Generated {len(minimal_samples)} minimal samples from CSV data")
                        break
            
            # If no CSV data, create generic samples
            if not minimal_samples:
                generic_samples = [
                    {
                        'user_input': "What is the general approach to troubleshooting?",
                        'reference': "The general approach involves identifying the problem, gathering information, and applying systematic solutions.",
                        'reference_contexts': ["General troubleshooting principles"],
                        'eval_sample': {
                            'user_input': "What is the general approach to troubleshooting?",
                            'reference': "The general approach involves identifying the problem, gathering information, and applying systematic solutions.",
                            'reference_contexts': ["General troubleshooting principles"]
                        },
                        'synthesizer_name': 'generic_fallback_generator'
                    },
                    {
                        'user_input': "How do you ensure system reliability?",
                        'reference': "System reliability is ensured through regular monitoring, maintenance, and following best practices.",
                        'reference_contexts': ["System reliability best practices"],
                        'eval_sample': {
                            'user_input': "How do you ensure system reliability?",
                            'reference': "System reliability is ensured through regular monitoring, maintenance, and following best practices.",
                            'reference_contexts': ["System reliability best practices"]
                        },
                        'synthesizer_name': 'generic_fallback_generator'
                    }
                ]
                
                minimal_samples.extend(generic_samples)
                logger.info(f"✅ Generated {len(minimal_samples)} generic fallback samples")
        
        except Exception as e:
            logger.error(f"❌ Minimal testset generation failed: {e}")
        
        return minimal_samples
    
    def _run_post_execution_validation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run post-execution validation to ensure pipeline outputs are valid."""
        
        logger.info("🔍 Running post-execution validation...")
        
        validation_results = {
            'success': True,
            'output_validation': {},
            'quality_metrics': {},
            'recommendations': [],
            'issues_found': []
        }
        
        try:
            # Validate testset outputs
            if 'testset_generation' in results and results['testset_generation']['success']:
                testset_validation = self._validate_testset_outputs(results['testset_generation'])
                validation_results['output_validation']['testset'] = testset_validation
                
                if not testset_validation['success']:
                    validation_results['success'] = False
                    validation_results['issues_found'].extend(testset_validation.get('issues', []))
            
            # Validate evaluation outputs
            if 'evaluation' in results and results['evaluation']['success']:
                eval_validation = self._validate_evaluation_outputs(results['evaluation'])
                validation_results['output_validation']['evaluation'] = eval_validation
                
                if not eval_validation['success']:
                    validation_results['success'] = False
                    validation_results['issues_found'].extend(eval_validation.get('issues', []))
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_overall_quality_metrics(results)
            validation_results['quality_metrics'] = quality_metrics
            
            # Generate recommendations
            recommendations = self._generate_pipeline_recommendations(results, validation_results)
            validation_results['recommendations'] = recommendations
            
            logger.info(f"✅ Post-execution validation complete: {len(validation_results['issues_found'])} issues found")
            
        except Exception as e:
            validation_results['success'] = False
            validation_results['error'] = str(e)
            logger.error(f"❌ Post-execution validation failed: {e}")
        
        return validation_results
    
    def _validate_testset_outputs(self, testset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate testset generation outputs."""
        
        validation = {
            'success': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Check if testset data exists
            if 'testset_data' not in testset_results:
                validation['issues'].append("No testset_data found in results")
                validation['success'] = False
                return validation
            
            samples = testset_results['testset_data']
            validation['metrics']['total_samples'] = len(samples)
            
            # Check sample quality
            if len(samples) == 0:
                validation['issues'].append("Testset is empty")
                validation['success'] = False
            elif len(samples) < 3:
                validation['issues'].append(f"Testset is very small: {len(samples)} samples")
            
            # Check sample structure
            required_fields = ['user_input', 'reference']
            valid_structure_count = 0
            
            for i, sample in enumerate(samples[:10]):  # Check first 10 samples
                missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
                if missing_fields:
                    validation['issues'].append(f"Sample {i} missing required fields: {missing_fields}")
                else:
                    valid_structure_count += 1
            
            validation['metrics']['structure_validity_rate'] = valid_structure_count / min(10, len(samples))
            
            if validation['metrics']['structure_validity_rate'] < 0.8:
                validation['success'] = False
                validation['issues'].append("Low sample structure validity rate")
            
            logger.info(f"📊 Testset validation: {len(samples)} samples, {validation['metrics']['structure_validity_rate']:.1%} structure validity")
            
        except Exception as e:
            validation['success'] = False
            validation['error'] = str(e)
        
        return validation
    
    def _validate_evaluation_outputs(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evaluation outputs."""
        
        validation = {
            'success': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Check basic metrics
            total_questions = eval_results.get('total_questions', 0)
            successful_evaluations = eval_results.get('successful_evaluations', 0)
            
            validation['metrics']['total_questions'] = total_questions
            validation['metrics']['successful_evaluations'] = successful_evaluations
            
            if total_questions == 0:
                validation['issues'].append("No questions were evaluated")
                validation['success'] = False
            
            success_rate = successful_evaluations / total_questions if total_questions > 0 else 0
            validation['metrics']['evaluation_success_rate'] = success_rate
            
            if success_rate < 0.5:
                validation['issues'].append(f"Low evaluation success rate: {success_rate:.1%}")
                validation['success'] = False
            
            # Check output files exist
            output_files = ['detailed_results_file', 'summary_report_file', 'csv_report_file']
            missing_files = []
            
            for file_key in output_files:
                if file_key in eval_results:
                    file_path = Path(eval_results[file_key])
                    if not file_path.exists():
                        missing_files.append(file_key)
            
            if missing_files:
                validation['issues'].append(f"Missing output files: {missing_files}")
                validation['success'] = False
            
            logger.info(f"📊 Evaluation validation: {success_rate:.1%} success rate, {len(missing_files)} missing files")
            
        except Exception as e:
            validation['success'] = False
            validation['error'] = str(e)
        
        return validation
    
    def _calculate_overall_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall pipeline quality metrics."""
        
        metrics = {
            'data_quality_score': 0.0,
            'pipeline_success_rate': 0.0,
            'validation_success_rate': 0.0,
            'overall_quality_score': 0.0
        }
        
        try:
            # Calculate data quality from CSV validation
            csv_reports = self.validation_reports.get('csv_validation', [])
            if csv_reports:
                avg_data_quality = sum(report.get('data_quality_score', 0.0) for report in csv_reports) / len(csv_reports)
                metrics['data_quality_score'] = avg_data_quality
            else:
                metrics['data_quality_score'] = 1.0  # Assume good if no CSV validation
            
            # Calculate pipeline success rate
            completed_stages = len(results.get('stages_completed', []))
            total_stages = 3  # testset-generation, evaluation, reporting (keyword-extraction removed)
            metrics['pipeline_success_rate'] = completed_stages / total_stages
            
            # Calculate validation success rate
            sample_reports = self.validation_reports.get('sample_validation', [])
            if sample_reports:
                avg_validation_success = sum(report.get('success_rate', 0.0) for report in sample_reports) / len(sample_reports)
                metrics['validation_success_rate'] = avg_validation_success
            else:
                metrics['validation_success_rate'] = 1.0
            
            # Calculate overall quality score (weighted average)
            weights = {
                'data_quality_score': 0.3,
                'pipeline_success_rate': 0.4,
                'validation_success_rate': 0.3
            }
            
            metrics['overall_quality_score'] = sum(
                metrics[key] * weight for key, weight in weights.items()
            )
            
            logger.info(f"📊 Overall quality metrics: {metrics['overall_quality_score']:.1%} overall quality")
            
        except Exception as e:
            logger.error(f"❌ Quality metrics calculation failed: {e}")
        
        return metrics
    
    def _generate_pipeline_recommendations(self, results: Dict[str, Any], validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for pipeline improvement."""
        
        recommendations = []
        
        try:
            # Analyze quality metrics
            quality_metrics = validation_results.get('quality_metrics', {})
            overall_quality = quality_metrics.get('overall_quality_score', 0.0)
            
            if overall_quality < 0.7:
                recommendations.append("Consider improving data quality through better preprocessing")
            
            if overall_quality < 0.5:
                recommendations.append("Pipeline quality is low - review configuration and data sources")
            
            # Analyze validation success rates
            validation_success = quality_metrics.get('validation_success_rate', 0.0)
            if validation_success < 0.8:
                recommendations.append("High sample rejection rate - consider adjusting validation criteria or improving data quality")
            
            # Analyze pipeline success
            pipeline_success = quality_metrics.get('pipeline_success_rate', 0.0)
            if pipeline_success < 1.0:
                recommendations.append("Some pipeline stages failed - check logs for specific issues")
            
            # Analyze testset generation
            if 'testset_generation' in results:
                tg_results = results['testset_generation']
                if tg_results.get('samples_rejected', 0) > tg_results.get('samples_validated', 1):
                    recommendations.append("Many samples were rejected during validation - improve data preprocessing")
            
            # Performance recommendations
            if 'total_duration' in results and results['total_duration'] > 3600:  # More than 1 hour
                recommendations.append("Pipeline took significant time - consider optimizing configuration or using smaller datasets for testing")
            
            logger.info(f"📋 Generated {len(recommendations)} recommendations for pipeline improvement")
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
        
        return recommendations
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        
        summary = {
            'csv_validation_summary': self.csv_validator.get_validation_summary(),
            'sample_validation_summary': self.sample_validator.get_validation_summary(),
            'kg_validation_summary': self.kg_validator.get_validation_summary(),
            'overall_quality_score': 0.0,
            'total_issues_found': 0,
            'total_fixes_applied': 0,
            'recommendations_count': 0
        }
        
        try:
            # Aggregate statistics
            csv_summary = summary['csv_validation_summary']
            sample_summary = summary['sample_validation_summary']
            kg_summary = summary['kg_validation_summary']
            
            # Calculate overall quality score
            quality_scores = []
            
            if csv_summary.get('overall_preservation_rate'):
                quality_scores.append(csv_summary['overall_preservation_rate'])
            
            if sample_summary.get('overall_preservation_rate'):
                quality_scores.append(sample_summary['overall_preservation_rate'])
            
            if kg_summary.get('overall_data_quality'):
                quality_scores.append(kg_summary['overall_data_quality'])
            
            if quality_scores:
                summary['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
            
            # Count total issues and fixes
            summary['total_issues_found'] = (
                csv_summary.get('total_issues_found', 0) +
                sample_summary.get('total_validation_errors', 0) +
                kg_summary.get('issues_found', 0)
            )
            
            summary['total_fixes_applied'] = (
                csv_summary.get('total_fixes_applied', 0) +
                kg_summary.get('fixes_applied', 0)
            )
            
            logger.info(f"📊 Validation summary: {summary['overall_quality_score']:.1%} quality, {summary['total_fixes_applied']} fixes applied")
            
        except Exception as e:
            logger.error(f"❌ Validation summary generation failed: {e}")
        
        return summary
    
    def _save_validation_reports(self, validation_reports: Dict[str, Any]) -> Path:
        """Save comprehensive validation reports."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"comprehensive_validation_report_{self.run_id}_{timestamp}.json"
            report_path = self.output_dirs['base'] / "validation_reports" / report_filename
            
            # Ensure directory exists
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to reports
            comprehensive_report = {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'pipeline_config': self.config,
                'validation_reports': validation_reports,
                'validator_summaries': {
                    'csv_validator': self.csv_validator.get_validation_summary(),
                    'sample_validator': self.sample_validator.get_validation_summary(),
                    'kg_validator': self.kg_validator.get_validation_summary()
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Saved comprehensive validation report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save validation reports: {e}")
            # Return a dummy path
            return self.output_dirs['base'] / "validation_report_save_failed.txt"
    
    def _analyze_pipeline_error(self, error: Exception, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline errors and provide insights."""
        
        analysis = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'likely_causes': [],
            'suggested_fixes': [],
            'stages_completed': results.get('stages_completed', []),
            'validation_issues': []
        }
        
        try:
            error_message = str(error).lower()
            
            # Analyze common error patterns
            if 'nan' in error_message or 'validation' in error_message:
                analysis['likely_causes'].append("Data quality issues with NaN values")
                analysis['suggested_fixes'].append("Run CSV validation and cleaning before pipeline execution")
            
            if 'memory' in error_message or 'out of memory' in error_message:
                analysis['likely_causes'].append("Insufficient memory for large dataset processing")
                analysis['suggested_fixes'].append("Reduce batch size or use smaller datasets")
            
            if 'timeout' in error_message or 'connection' in error_message:
                analysis['likely_causes'].append("Network or API connectivity issues")
                analysis['suggested_fixes'].append("Check API endpoints and network connectivity")
            
            if 'file not found' in error_message or 'no such file' in error_message:
                analysis['likely_causes'].append("Missing input files or incorrect file paths")
                analysis['suggested_fixes'].append("Verify all file paths in configuration")
            
            # Analyze validation reports for additional insights
            validation_reports = results.get('validation_reports', {})
            for stage, report in validation_reports.items():
                if isinstance(report, dict) and not report.get('success', True):
                    analysis['validation_issues'].append(f"{stage}: {report.get('error', 'Unknown validation issue')}")
            
            logger.info(f"📊 Error analysis complete: {len(analysis['likely_causes'])} causes identified")
            
        except Exception as e:
            logger.error(f"❌ Error analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _find_knowledge_graph_files(self) -> List[Path]:
        """Find knowledge graph files in the workspace."""
        
        kg_files = []
        
        try:
            # Look in configuration
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            existing_kg_file = kg_config.get('existing_kg_file')
            
            if existing_kg_file:
                kg_path = Path(existing_kg_file)
                if not kg_path.is_absolute():
                    kg_path = Path(self.config.get('output', {}).get('base_dir', './outputs')).parent / existing_kg_file
                
                if kg_path.exists():
                    kg_files.append(kg_path)
            
            # Look in standard locations
            search_dirs = [
                self.output_dirs.get('base', Path('./outputs')),
                Path('./data'),
                Path('./knowledge_graphs')
            ]
            
            for search_dir in search_dirs:
                if search_dir.exists():
                    kg_files.extend(search_dir.glob('*.json'))
                    kg_files.extend(search_dir.glob('**/*.json'))
            
            # Filter to likely KG files
            kg_files = [f for f in kg_files if any(keyword in f.name.lower() for keyword in ['kg', 'knowledge', 'graph'])]
            
            # Remove duplicates
            kg_files = list(set(kg_files))
            
        except Exception as e:
            logger.warning(f"⚠️ Error finding KG files: {e}")
        
        return kg_files
    
    def _update_config_with_cleaned_file(self, original_path: str, cleaned_path: str):
        """Update configuration to use cleaned files."""
        
        try:
            # Update CSV files
            csv_files = self.config.get('input', {}).get('csv_files', [])
            if original_path in csv_files:
                index = csv_files.index(original_path)
                csv_files[index] = cleaned_path
                logger.info(f"🔧 Updated config: {original_path} -> {cleaned_path}")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to update config: {e}")
