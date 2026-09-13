param(
    [string]$ContentPath = 'docs/paper/v5/introduction/slides.json',
    [string]$BuildPath = 'outputs/presentation-build/yr315-v5-intro',
    [string]$FinalPath = 'docs/paper/v5/introduction/v5_졸업논문_서론.pptx'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$workspaceRoot = (Get-Location).Path
$contentFile = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $ContentPath))
$buildDirectory = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $BuildPath))
$finalFile = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $FinalPath))
if (Test-Path -LiteralPath $finalFile) { throw "Output exists: $finalFile" }
[void](New-Item -ItemType Directory -Path $buildDirectory -Force)
$candidateFile = Join-Path $buildDirectory 'candidate.pptx'
if (Test-Path -LiteralPath $candidateFile) { throw "Draft exists: $candidateFile" }
$deck = Get-Content -LiteralPath $contentFile -Raw -Encoding UTF8 | ConvertFrom-Json
$font = '맑은 고딕'
$navy = 0x49341C
$gray = 0x666666
$issues = [Collections.Generic.List[object]]::new()
$shapeRecords = [Collections.Generic.List[object]]::new()

function Add-Text($slide, [string]$name, [string]$text, [double]$x, [double]$y,
                  [double]$width, [double]$height, [double]$size,
                  [bool]$bold = $false, [int]$color = 0x222222) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $width, $height)
    $shape.Name = $name
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.TextRange.Text = $text.Replace("`n", "`r")
    $range = $shape.TextFrame.TextRange
    $range.Font.Name = $font
    $range.Font.NameFarEast = $font
    $range.Font.Size = $size
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $range.Font.Color.RGB = $color
    $range.ParagraphFormat.Bullet.Visible = 0
    $range.ParagraphFormat.SpaceAfter = 10
    $range.ParagraphFormat.LineRuleWithin = -1
    $range.ParagraphFormat.SpaceWithin = 1.13
    $shape.TextFrame2.AutoSize = 0
    $shape.Width = [float]$width
    $shape.Height = [float]$height
    return $shape
}

function Find-Ref([int]$referenceId) {
    return @($deck.references | Where-Object id -EQ $referenceId)[0]
}

function Add-References($slide, $item) {
    if (@($item.refs).Count -eq 0) { return }
    $compact = $item.PSObject.Properties.Name -contains 'compact_refs'
    $index = 0
    foreach ($referenceId in $item.refs) {
        $ref = Find-Ref $referenceId
        if ($compact) {
            $label = "[$($ref.id)] $($ref.authors.Split(',')[0]) 등 ($($ref.year))"
            $x = 48 + ($index % 3) * 288
            $y = 471 + [math]::Floor($index / 3) * 25
            $shape = Add-Text $slide "reference-$referenceId" $label $x $y 275 23 12.5 $false $gray
        } else {
            $label = "[$($ref.id)] $($ref.title) ($($ref.year))"
            $shape = Add-Text $slide "reference-$referenceId" $label 48 (453 + $index * 36) 850 35 12.5 $false $gray
        }
        $shape.TextFrame.TextRange.ActionSettings.Item(1).Hyperlink.Address = $ref.url
        $index++
    }
}

function Check-Shape($slide, $shape, [string]$label) {
    if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
        $range = $shape.TextFrame2.TextRange
        $availableHeight = $shape.Height - $shape.TextFrame2.MarginTop - $shape.TextFrame2.MarginBottom
        $availableWidth = $shape.Width - $shape.TextFrame2.MarginLeft - $shape.TextFrame2.MarginRight
        $record = [ordered]@{slide=$slide.SlideIndex; shape=$label; width=$shape.Width;
            height=$shape.Height; text_width=$range.BoundWidth; text_height=$range.BoundHeight}
        $shapeRecords.Add($record)
        if ($range.BoundHeight -gt $availableHeight + 2 -or $range.BoundWidth -gt $availableWidth + 3) {
            $issues.Add($record)
        }
    }
}

$application = New-Object -ComObject PowerPoint.Application
$initialPresentationCount = $application.Presentations.Count
$presentation = $null
$reopened = $null
try {
    $presentation = $application.Presentations.Add(0)
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540
    foreach ($item in $deck.slides) {
        $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.Solid()
        $slide.Background.Fill.ForeColor.RGB = 0xFFFFFF
        $kind = if ($item.PSObject.Properties.Name -contains 'kind') { $item.kind } else { 'content' }
        if ($kind -eq 'cover') {
            [void](Add-Text $slide 'title' $item.title 48 105 864 225 42 $true $navy)
            [void](Add-Text $slide 'subtitle' $item.subtitle 48 350 864 110 20 $false $gray)
        } elseif ($kind -eq 'references') {
            [void](Add-Text $slide 'title' $item.title 48 37 864 62 32 $true $navy)
            $count = @($item.reference_ids).Count
            $entryHeight = if ($count -eq 4) { 99 } else { 131 }
            $index = 0
            foreach ($referenceId in $item.reference_ids) {
                $ref = Find-Ref $referenceId
                $text = "[$($ref.id)] $($ref.authors) ($($ref.year)).`r$($ref.title).`r$($ref.venue).`r$($ref.url)"
                $shape = Add-Text $slide "bibliography-$referenceId" $text 48 (112+$index*$entryHeight) 864 ($entryHeight-6) 14.5
                $shape.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 1
                $shape.TextFrame.TextRange.ParagraphFormat.SpaceWithin = 1.0
                $shape.TextFrame.TextRange.Paragraphs(2).Font.Size = 16
                $shape.TextFrame.TextRange.Paragraphs(4).Font.Size = 12.5
                $shape.TextFrame.TextRange.Paragraphs(4).Font.Color.RGB = $gray
                $shape.TextFrame.TextRange.Paragraphs(4).ActionSettings.Item(1).Hyperlink.Address = $ref.url
                $index++
            }
        } else {
            [void](Add-Text $slide 'title' $item.title 48 37 864 65 32 $true $navy)
            if ($item.PSObject.Properties.Name -contains 'table') {
                $rowCount = @($item.table.rows).Count + 1
                $tableShape = $slide.Shapes.AddTable($rowCount, @($item.table.headers).Count, 48, 127, 864, 248)
                $tableShape.Name = 'content-table'
                $table = $tableShape.Table
                for ($column = 1; $column -le $table.Columns.Count; $column++) {
                    $table.Columns.Item($column).Width = $item.table.widths[$column-1]
                }
                for ($row = 1; $row -le $rowCount; $row++) {
                    $table.Rows.Item($row).Height = [float]$(if ($row -eq 1) { 38 } else { 210 / ($rowCount-1) })
                    for ($column = 1; $column -le $table.Columns.Count; $column++) {
                        $cell = $table.Cell($row, $column).Shape
                        $text = if ($row -eq 1) { $item.table.headers[$column-1] } else { $item.table.rows[$row-2][$column-1] }
                        $cell.TextFrame.TextRange.Text = $text.Replace("`n", "`r")
                        $cell.TextFrame.MarginLeft = 10
                        $cell.TextFrame.MarginRight = 8
                        $cell.TextFrame.MarginTop = 5
                        $cell.TextFrame.MarginBottom = 5
                        $cell.TextFrame.VerticalAnchor = 3
                        $range = $cell.TextFrame.TextRange
                        $range.Font.Name = $font
                        $range.Font.NameFarEast = $font
                        $range.Font.Size = [float]$(if ($row -eq 1) { 18 } else { 18.5 })
                        $range.Font.Bold = $(if ($row -eq 1) { -1 } else { 0 })
                        $range.Font.Color.RGB = 0x222222
                        $range.ParagraphFormat.SpaceAfter = 0
                        $range.ParagraphFormat.LineRuleWithin = -1
                        $range.ParagraphFormat.SpaceWithin = 1.0
                        $cell.Fill.Solid()
                        $cell.Fill.ForeColor.RGB = $(if ($row -eq 1) { 0xEAEAEA } else { 0xFFFFFF })
                        for ($border = 1; $border -le 4; $border++) {
                            $table.Cell($row, $column).Borders.Item($border).ForeColor.RGB = 0xCFCFCF
                            $table.Cell($row, $column).Borders.Item($border).Weight = 0.5
                        }
                    }
                }
            } else {
                $isBullets = $item.PSObject.Properties.Name -contains 'bullets'
                $lines = if ($isBullets) { $item.bullets } else { $item.paragraphs }
                $shape = Add-Text $slide 'body' ($lines -join "`r") 48 132 864 249 23
                $shape.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 22
                if ($isBullets) {
                    $shape.TextFrame.TextRange.ParagraphFormat.Bullet.Visible = -1
                    $shape.TextFrame.TextRange.ParagraphFormat.Bullet.Character = 8226
                    $shape.TextFrame.Ruler.Levels.Item(1).FirstMargin = 0
                    $shape.TextFrame.Ruler.Levels.Item(1).LeftMargin = 20
                }
            }
            [void](Add-Text $slide 'takeaway' $item.takeaway 48 399 864 58 20 $true $navy)
            Add-References $slide $item
        }
        [void](Add-Text $slide 'page' ([string]$slide.SlideIndex) 921 513 26 16 10.5 $false $gray)
        $notes = $item.notes
        foreach ($referenceId in $item.refs) {
            $ref = Find-Ref $referenceId
            $notes += "`r`r[$($ref.id)] $($ref.authors) ($($ref.year)). $($ref.title). $($ref.venue).`r$($ref.url)`r확인 자료: $($ref.verified_source)`r인용 범위: $($ref.scope)"
        }
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes
    }
    $presentation.SaveAs($candidateFile, 24)
    $presentation.Close()
    $presentation = $null
    $reopened = $application.Presentations.Open($candidateFile, -1, 0, 0)
    if ($reopened.Slides.Count -ne 19) { throw 'Unexpected slide count' }
    $tableSlides = [Collections.Generic.List[int]]::new()
    foreach ($slide in $reopened.Slides) {
        foreach ($shape in $slide.Shapes) {
            Check-Shape $slide $shape $shape.Name
            if ($shape.HasTable -eq -1) {
                $tableSlides.Add($slide.SlideIndex)
                for ($row = 1; $row -le $shape.Table.Rows.Count; $row++) {
                    for ($column = 1; $column -le $shape.Table.Columns.Count; $column++) {
                        Check-Shape $slide $shape.Table.Cell($row,$column).Shape "table-$row-$column"
                    }
                }
            }
        }
        $slide.Export((Join-Path $buildDirectory ('slide-{0:D2}.png' -f $slide.SlideIndex)), 'PNG', 1600, 900)
    }
    $qa = [ordered]@{slides=$reopened.Slides.Count; powerpoint_version=$application.Version;
        roundtrip_opened=$true; rendered_all_slides=$true; editable_table_slides=@($tableSlides);
        text_fit_issues=@($issues); shape_geometry=@($shapeRecords); content_sha256=(Get-FileHash -Algorithm SHA256 $contentFile).Hash.ToLower()}
    $qa | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $buildDirectory 'powerpoint-check.json') -Encoding UTF8
    if ($issues.Count -gt 0) { throw "Text fit issues: $($issues.Count), inspect powerpoint-check.json" }
    $reopened.SaveCopyAs($finalFile, 24)
    Write-Output "CREATED: $finalFile"
    Write-Output "Slides: $($reopened.Slides.Count), native tables: $($tableSlides.Count), text fit issues: 0"
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $reopened) { $reopened.Close() }
    if ($initialPresentationCount -eq 0 -and $application.Presentations.Count -eq 0) { $application.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
}
