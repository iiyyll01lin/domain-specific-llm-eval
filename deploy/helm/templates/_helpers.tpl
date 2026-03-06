{{/*
_helpers.tpl — shared template helpers
*/}}

{{- define "llm-eval.fullname" -}}
{{- printf "%s-%s" .Release.Name .name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "llm-eval.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "llm-eval.selectorLabels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "llm-eval.image" -}}
{{- $registry := .global.imageRegistry | default "" -}}
{{- printf "%s%s:%s" $registry .repository (.tag | default "latest") }}
{{- end }}
