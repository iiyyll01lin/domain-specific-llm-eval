# PowerShell script to convert CSV to Excel format
# This creates the exact Excel file format expected by the evaluation pipeline

$csvPath = "SystemQAListallQuestion_eval_step4_final_report 1.csv"
$excelPath = "SystemQAListallQuestion_eval_step4_final_report 1.xlsx"

# Create Excel application object
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false

try {
    # Open CSV file
    $workbook = $excel.Workbooks.Open((Resolve-Path $csvPath).Path)
    
    # Save as Excel format
    $workbook.SaveAs((Join-Path (Get-Location) $excelPath), 51) # 51 = xlOpenXMLWorkbook (.xlsx)
    
    Write-Host "✅ Successfully converted CSV to Excel format: $excelPath"
    Write-Host "📊 Dataset contains 10 synthetic samples ready for evaluation"
    
} catch {
    Write-Host "❌ Error converting CSV to Excel: $($_.Exception.Message)"
    Write-Host "💡 You can manually open the CSV file in Excel and save as .xlsx"
} finally {
    # Clean up
    if ($workbook) { $workbook.Close($false) }
    if ($excel) { $excel.Quit() }
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
}

Write-Host ""
Write-Host "🎯 Dataset is ready! You can now run the evaluation pipeline:"
Write-Host "   1. python contextual_keyword_gate.py"
Write-Host "   2. python dynamic_ragas_gate_with_human_feedback.py"
