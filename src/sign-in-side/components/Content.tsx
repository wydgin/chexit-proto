import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { Link as RouterLink } from 'react-router-dom';
import AutoFixHighRoundedIcon from '@mui/icons-material/AutoFixHighRounded';
import ConstructionRoundedIcon from '@mui/icons-material/ConstructionRounded';
import SettingsSuggestRoundedIcon from '@mui/icons-material/SettingsSuggestRounded';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';
import { SitemarkIcon } from './CustomIcons';

const items = [
  {
    icon: <SettingsSuggestRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Adaptable detection',
    description:
      'Our AI adapts to diverse local and international chest X-rays, boosting TB sensitivity while keeping workflows fast.',
  },
  {
    icon: <ConstructionRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Built for real hospitals',
    description:
      'Lightweight models and optimized preprocessing let it run reliably on modest hardware for long-term use.',
  },
  {
    icon: <ThumbUpAltRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Clinician-centered',
    description:
      'Radiologists get clear TB risk scores, intuitive heatmaps, and brief summaries that fit easily into existing routines.',
  },
  {
    icon: <AutoFixHighRoundedIcon sx={{ color: 'text.secondary' }} />,
    title: 'Assistive and explainable',
    description:
      'CLAHE, U-Net lung segmentation, and an ensemble of CNNs deliver robust, interpretable predictions for TB care in the Philippines.',
  },
];

export default function Content() {
  return (
    <Stack
      sx={{ flexDirection: 'column', alignSelf: 'center', gap: 4, maxWidth: 450 }}
    >
      <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
        <SitemarkIcon />
      </Box>
      {items.map((item, index) => (
        <Stack key={index} direction="row" sx={{ gap: 2 }}>
          {item.icon}
          <div>
            <Typography gutterBottom sx={{ fontWeight: 'medium' }}>
              {item.title}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              {item.description}
            </Typography>
          </div>
        </Stack>
      ))}
      <Button
        component={RouterLink}
        to="/dashboard"
        variant="contained"
        color="primary"
        sx={{ alignSelf: { xs: 'stretch', md: 'flex-start' }, mt: 1 }}
      >
        Try the analyzer — upload &amp; get a heatmap
      </Button>
    </Stack>
  );
}
