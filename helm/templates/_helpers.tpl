{{- define "arena.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "arena.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "arena.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: crypto-research
{{- end }}

{{- define "arena.selectorLabels" -}}
app.kubernetes.io/name: {{ include "arena.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "arena.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/* PSA-restricted pod security context */}}
{{- define "arena.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/* PSA-restricted container security context */}}
{{- define "arena.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end }}

{{/* Common env: from values + fixed paths */}}
{{- define "arena.env" -}}
{{- range $k, $v := .root.Values.env }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
- name: UNIVERSE_PATH
  value: {{ .universe | default "/app/config/universe.yaml" }}
- name: HOME
  value: /tmp
{{- end }}

{{/*
Pod template shared by every Job/CronJob.
Usage: include "arena.podTemplate" (dict "root" $ "component" "tick" "args" (list "tick") "resources" $.Values.resources.tick)
*/}}
{{- define "arena.podTemplate" -}}
metadata:
  labels:
    {{- include "arena.labels" .root | nindent 4 }}
    app.kubernetes.io/component: {{ .component }}
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  securityContext:
    {{- include "arena.podSecurityContext" .root | nindent 4 }}
  containers:
    - name: arena
      image: {{ include "arena.image" .root }}
      imagePullPolicy: {{ .root.Values.image.pullPolicy }}
      args: {{ .args | toJson }}
      envFrom:
        - secretRef:
            name: {{ .root.Values.secrets.targetName }}
      env:
        {{- include "arena.env" (dict "root" .root "universe" .universe) | nindent 8 }}
      resources:
        {{- toYaml .resources | nindent 8 }}
      securityContext:
        {{- include "arena.containerSecurityContext" .root | nindent 8 }}
      volumeMounts:
        - name: tmp
          mountPath: /tmp
  volumes:
    - name: tmp
      emptyDir: {}
{{- end }}
