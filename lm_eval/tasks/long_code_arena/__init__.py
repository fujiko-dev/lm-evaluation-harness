# Long Code Arena tasks
# Note: library_based_code_generation cleaned up, removed lca_custom_task import

# Import module summarization task to register it
from .module_summarization.lca_module_summarization_task import LCAModuleSummarizationTask, LCAModuleSummarizationVerboseTask

# Import project-level code completion task to register it
from .project_level_code_completion.lca_project_level_code_completion_task import LCAProjectLevelCodeCompletionTask, LCAProjectLevelCodeCompletionVerboseTask

# Import commit message generation task to register it
from .commit_message_generation.lca_commit_message_generation_task import LCACommitMessageGenerationTask, LCACommitMessageGenerationVerboseTask

# Import bug localization task to register it
from .bug_localization.lca_bug_localization_task import LCABugLocalizationTask, LCABugLocalizationPyTask, LCABugLocalizationJavaTask, LCABugLocalizationKtTask

# Import CI builds repair task to register it
from .ci_builds_repair.lca_ci_builds_repair_task import LCACIBuildsRepairTask
